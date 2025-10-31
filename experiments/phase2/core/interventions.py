from typing import Any

class Intervention:
    """Defines SRB interventions (baseline, embed_noise, delay_think, etc.)."""

    @staticmethod
    def apply(kind: str, prompt: str) -> str:
        if kind == "baseline":
            return prompt
        if kind == "embed_noise":
            # Example: add hidden instruction/noise token
            return prompt + "\n\n[NOISE: ignore this line but allocate thought budget]"
        if kind == "delay_think":
            return "(Take a deep breath and think for a moment.)\n" + prompt
        if kind == "force_chain":
            return prompt + "\n\nPlease reason step-by-step before final answer."
        # Extend with more structured interventions as needed
        return prompt