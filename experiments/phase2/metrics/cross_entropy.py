class CrossEntropySurprise:
    """Compute cross-entropy (negative log-likelihood) of a response under a
    baseline model. Stub exposes a hook to your tokenizer/model.
    """
    def __init__(self, baseline_model_name: str):
        self.name = baseline_model_name
        self._hf_ok = False
        self._tok = None
        self._lm = None
        self._device = "cpu"

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            # Prefer MPS on Apple Silicon if available
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"

            self._tok = AutoTokenizer.from_pretrained(self.name)
            self._lm = AutoModelForCausalLM.from_pretrained(self.name)

            # Try moving to MPS; if it fails, fall back to CPU
            try:
                self._lm.to(self._device)
                print(f"[SRB][HF] CrossEntropySurprise using HF model: {self.name} (device={self._device})")
            except Exception as move_err:
                print(f"[SRB][HF] NLL model could not use {self._device}, falling back to CPU: {move_err}")
                self._device = "cpu"
                self._lm.to(self._device)
            print(f"[SRB][HF] CrossEntropySurprise active device: {self._device}")
            self._lm.eval()
            self._hf_ok = True
            print(f"[SRB][HF] CrossEntropySurprise using HF model: {self.name} (device={self._device})")
        except Exception as e:
            print(f"[SRB][HF] CrossEntropySurprise fallback (stub NLL): {e}")

    def nll(self, response_text: str) -> float:
        # TODO: implement with HF model; return per-token or total NLL
        return float(len(response_text)) * 0.01  # placeholder scaling
