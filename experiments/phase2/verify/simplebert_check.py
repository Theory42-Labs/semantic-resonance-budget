class SimpleBERTVerifier:
    """External low-vocab BERT-style sanity checker over coherence/similarity.
    Keep this OUTSIDE the main generation loop to avoid tool/data coupling.
    """
    def __init__(self, model_name: str):
        self.name = model_name
        self._hf_ok = False
        self._tok = None
        self._enc = None
        self._device = "cpu"

        try:
            from transformers import AutoTokenizer, AutoModel
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"

            self._tok = AutoTokenizer.from_pretrained(self.name)
            self._enc = AutoModel.from_pretrained(self.name)

            try:
                self._enc.to(self._device)
                print(f"[SRB][HF] Verifier using HF encoder: {self.name} (device={self._device})")
            except Exception as move_err:
                print(f"[SRB][HF] Verifier could not use {self._device}, falling back to CPU: {move_err}")
                self._device = "cpu"
                self._enc.to(self._device)
            print(f"[SRB][HF] Verifier active device: {self._device}")
            self._enc.eval()
            self._hf_ok = True
            print(f"[SRB][HF] Verifier using HF encoder: {self.name} (device={self._device})")
        except Exception as e:
            print(f"[SRB][HF] Verifier fallback (stub similarity): {e}")

    def similarity(self, a: str, b: str) -> float:
        # TODO: cosine over [CLS] or pooled outputs
        return 0.5  # placeholder