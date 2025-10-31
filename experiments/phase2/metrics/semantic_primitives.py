class SemanticPrimitiveCoherence:
    """NSM primitive alignment checker.
    Counts presence/coverage/structure of primitive lexeme set as a cheap proxy.
    Extend to dependency/role alignment later.
    """
    def __init__(self, nsm_list_path: str):
        self.primitives = self._load(nsm_list_path)

    def _load(self, path: str) -> set[str]:
        try:
            return set(Path(path).read_text().split())
        except Exception:
            # Minimal seed list; replace with canonical NSM list in vendors/
            return {
                "I","YOU","SOMEONE","PEOPLE","SAY","THINK","KNOW","DO","HAPPEN",
                "BECAUSE","IF","GOOD","BAD","BIG","SMALL","MANY","FEW","BEFORE","AFTER",
                "HERE","NOW","THIS","SAME","OTHER"
            }

    def score(self, text: str) -> float:
        words = set(w.strip(".,!?;:" ).upper() for w in text.split())
        hits = words.intersection(self.primitives)
        # Simple coverage ratio as a starting point
        return len(hits) / max(1, len(self.primitives))