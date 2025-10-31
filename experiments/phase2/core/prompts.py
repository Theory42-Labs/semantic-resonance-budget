@dataclass
class PromptItem:
    bucket: str
    text: str

class PromptBank:
    def __init__(self):
        # TODO: load from file if desired
        self.prompts = [
            PromptItem("creative", "Write a micro-fable about a patient river."),
            PromptItem("factual", "Explain the greenhouse effect in two sentences."),
            PromptItem("reasoning", "If a train leaves at 2pm... reason about arrival time with constraints."),
        ]

    def sample(self, bucket: str) -> PromptItem:
        cand = [p for p in self.prompts if p.bucket == bucket]
        return random.choice(cand)