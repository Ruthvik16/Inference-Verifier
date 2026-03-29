import yaml
import numpy as np


class RuleBasedVerifier:
    """
    A pluggable rule-based verifier with feedback generation.

    Returns:
        {
            "status": "correct" | "incorrect" | "continue",
            "feedback": str
        }
    """

    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Logprob heuristics
        self.logprob_confidence_threshold = -2.0
        self.logprob_stable_window = 5

        # Keyword heuristics
        self.good_keywords = [
            "therefore",
            "thus",
            "in conclusion",
        ]

        self.bad_keywords = [
            "this is wrong",
            "incorrect",
            "mistake",
            "wait",
            "contradiction",
        ]

    # ----------------------------------------------------------------------
    # FEEDBACK GENERATORS (NEW)
    # ----------------------------------------------------------------------
    def _feedback_incorrect(self, partial_cot):
        """
        Generate corrective feedback when reasoning is incorrect.
        """
        return (
            "The reasoning contains an inconsistency or error. "
            "Re-evaluate the previous step carefully and correct the mistake. "
            "Ensure each step logically follows from the previous one."
        )

    def _feedback_correct(self, partial_cot):
        """
        Generate positive reinforcement when reasoning is correct.
        """
        return (
            "The reasoning so far appears correct and consistent. "
            "You may proceed to the final answer if sufficient."
        )

    def _feedback_continue(self, partial_cot):
        """
        Encourage continuation when reasoning is incomplete.
        """
        return (
            "Please Output Hi before you proceed"
        )

    # ----------------------------------------------------------------------
    # (A) Keyword heuristic
    # ----------------------------------------------------------------------
    def _keyword_signal(self, partial_cot):
        text = partial_cot.lower()

        for bad in self.bad_keywords:
            if bad in text:
                return "incorrect"

        for good in self.good_keywords:
            pattern = f" {good} "
            if pattern in text or text.startswith(f"{good} ") or text.endswith(f" {good}"):
                return "correct"

        return "continue"

    # ----------------------------------------------------------------------
    # (B) Logprob heuristic
    # ----------------------------------------------------------------------
    def _logprob_signal(self, logprobs):
        if not logprobs or len(logprobs) < self.logprob_stable_window:
            return "continue"

        window = logprobs[-self.logprob_stable_window:]
        mean_lp = np.mean(window)

        if mean_lp > self.logprob_confidence_threshold:
            return "correct"

        return "continue"

    # ----------------------------------------------------------------------
    # (C) Structure heuristic
    # ----------------------------------------------------------------------
    def _structure_signal(self, partial_cot):
        if any(x in partial_cot for x in ["1.", "2.", "3.", "- "]):
            return "correct"

        return "continue"

    # ----------------------------------------------------------------------
    # Unified evaluation
    # ----------------------------------------------------------------------
    def evaluate(self, partial_cot, logprobs=None):
        # """
        # Returns structured output:
        # {
        #     "status": ...,
        #     "feedback": ...
        # }
        # """

        # # -------------------------------
        # # 1. Keyword heuristic
        # # -------------------------------
        # verdict = self._keyword_signal(partial_cot)
        # if verdict != "continue":
        #     return {
        #         "status": verdict,
        #         "feedback": (
        #             self._feedback_incorrect(partial_cot)
        #             if verdict == "incorrect"
        #             else self._feedback_correct(partial_cot)
        #         )
        #     }

        # # -------------------------------
        # # 2. Logprob heuristic
        # # -------------------------------
        # if logprobs is not None:
        #     verdict = self._logprob_signal(logprobs)
        #     if verdict != "continue":
        #         return {
        #             "status": verdict,
        #             "feedback": (
        #                 self._feedback_correct(partial_cot)
        #                 if verdict == "correct"
        #                 else self._feedback_continue(partial_cot)
        #             )
        #         }

        # # -------------------------------
        # # 3. Structural heuristic
        # # -------------------------------
        # verdict = self._structure_signal(partial_cot)
        # if verdict != "continue":
        #     return {
        #         "status": verdict,
        #         "feedback": self._feedback_correct(partial_cot)
        #     }

        # -------------------------------
        # Default
        # -------------------------------
        return {
            "status": "continue",
            "feedback": self._feedback_continue(partial_cot)
        }