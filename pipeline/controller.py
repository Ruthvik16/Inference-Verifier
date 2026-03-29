import yaml
from llm_engine.cot_generator import CoTGenerator
from llm_engine.final_answer_generator import FinalAnswerGenerator
from verifier.rule_based_verifier import RuleBasedVerifier
from pipeline.early_exit_logic import EarlyExitLogic
from pipeline.utils import print_debug


class PipelineController:
    """
    Full reasoning pipeline controller.

    Workflow:
      1. Start generating chain-of-thought (CoT).
      2. Every N tokens → send partial CoT to verifier.
      3. Verifier returns: correct / incorrect / continue.
      4. EarlyExitLogic decides: exit / continue / abort.
      5. If exit → generate final concise answer.
      6. If continue → keep generating CoT.
      7. If abort → output fallback or request refinement.
    """

    def __init__(self, model, tokenizer, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Core modules
        self.cot_generator = CoTGenerator(model, tokenizer, config_path)
        self.final_answer_gen = FinalAnswerGenerator(model, tokenizer, config_path)
        self.verifier = RuleBasedVerifier(config_path)
        self.exit_logic = EarlyExitLogic(config_path)

        # Debug flags
        self.debug_cfg = self.config["debug"]

        # LLM generation settings
        self.checkpoint_interval = self.config["llm"]["checkpoint_interval"]
        self.max_cot_tokens = self.config["llm"]["max_cot_tokens"]

        
    def _inject_feedback(self, input_ids, feedback):
        feedback_text = f"""
    IMPORTANT:
    You made a mistake.

    Instruction:
    {feedback}

    You MUST follow this before continuing.

    Now continue:
    """

        feedback_ids = self.cot_generator.encode(feedback_text)["input_ids"]

        return torch.cat([input_ids, feedback_ids], dim=1)


    # ----------------------------------------------------------------------
    # MAIN EXECUTION ENTRY POINT
    # ----------------------------------------------------------------------
    def run(self, user_query):
        """
        Executes the full pipeline with:
        - sentence-level verification
        - inline corrective feedback injection
        """

        print_debug("Starting pipeline...", self.debug_cfg)

        full_cot = ""
        verifier_call_count = 0

        prompt = f"{user_query}\n"
        input_ids = None

        sentence_endings = [".", "?", "!"]

        for global_step in range(0, self.max_cot_tokens, self.checkpoint_interval):

            # ------------------------------------------------------
            # 1. Generate
            # ------------------------------------------------------
            source = prompt if input_ids is None else input_ids

            partial_text, logprobs, input_ids = \
                self.cot_generator.generate_until_checkpoint(source)

            full_cot += partial_text

            # ------------------------------------------------------
            # 2. Wait for logical statement completion
            # ------------------------------------------------------
            if not any(p in full_cot[-3:] for p in sentence_endings):
                prompt = None
                continue

            # ------------------------------------------------------
            # 3. Verifier
            # ------------------------------------------------------
            verifier_call_count += 1

            verifier_output = self.verifier.evaluate(full_cot, logprobs)
            status = verifier_output["status"]
            feedback = verifier_output.get("feedback", "")

            if self.debug_cfg["print_verifier_decision"]:
                print(f"[Verifier Status] {status}")
                print(f"[Verifier Feedback] {feedback}")

            # ------------------------------------------------------
            # 4. Decision
            # ------------------------------------------------------
            decision = self.exit_logic.decide(status, verifier_call_count)

            print("[DEBUG] Partial CoT:", partial_text)
            print("[DEBUG] Status:", status)
            print("[DEBUG] Decision:", decision)

            # ------------------------------------------------------
            # 5. Exit
            # ------------------------------------------------------
            if decision == "exit":
                print_debug("Early exit triggered...", self.debug_cfg)

                final_answer = self.final_answer_gen.generate_final_answer(full_cot)

                if self.debug_cfg["print_final_answer"]:
                    print(f"[Final Answer] {final_answer}")

                return final_answer

            # ------------------------------------------------------
            # 6. INLINE FEEDBACK INJECTION (NEW)
            # ------------------------------------------------------
            # if status == "incorrect":
            #     print_debug("Injecting inline correction...", self.debug_cfg)

            #     input_ids = self._inject_feedback(input_ids, feedback)

            #     # Also reflect in text (for debugging + final answer consistency)
            #     full_cot += f"\n[CORRECTION]: {feedback}\n"

            #     continue
            if status == "incorrect":
                guidance = f"""
            You MUST follow this:
            {feedback}
            """

                guidance_ids = tokenizer(guidance, return_tensors="pt")["input_ids"].to(model.device)

                # 🔥 concatenate BEFORE next forward pass
                input_ids = torch.cat([guidance_ids, input_ids], dim=1)

            # ------------------------------------------------------
            # 7. Continue normal generation
            # ------------------------------------------------------
            prompt = None

        # ----------------------------------------------------------
        # 8. Fallback
        # ----------------------------------------------------------
        print_debug("Max CoT reached...", self.debug_cfg)

        final_answer = self.final_answer_gen.generate_final_answer(full_cot)
        return final_answer