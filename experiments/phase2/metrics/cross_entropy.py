class CrossEntropySurprise:
    """Compute cross-entropy (negative log-likelihood) of a response under a
    baseline model. Stub exposes a hook to your tokenizer/model.
    """
    def __init__(self, baseline_model_name: str):
        self.model = None  # TODO: load small LM for NLL
        self.name = baseline_model_name

    def nll(self, response_text: str) -> float:
        # TODO: implement with HF model; return per-token or total NLL
        return float(len(response_text)) * 0.01  # placeholder scaling
