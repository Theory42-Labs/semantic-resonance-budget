class SimpleBERTVerifier:
    """External low-vocab BERT-style sanity checker over coherence/similarity.
    Keep this OUTSIDE the main generation loop to avoid tool/data coupling.
    """
    def __init__(self, model_name: str):
        self.name = model_name
        self.model = None  # TODO: load HF model/tokenizer

    def similarity(self, a: str, b: str) -> float:
        # TODO: cosine over [CLS] or pooled outputs
        return 0.5  # placeholder