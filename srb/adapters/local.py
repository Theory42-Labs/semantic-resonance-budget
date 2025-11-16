"""
Local adapter for Semantic Resonance Budget (SRB) using Hugging Face Transformers.

This adapter runs a causal language model locally (CPU / GPU / MPS) and
emits SRB-compatible TokenStep traces:

- token text
- logprob of the chosen token
- full probability distribution over the vocabulary
- per-token embedding (hidden state vector)

Requirements:
    pip install torch transformers
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

try:  # optional dependency
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "LocalTransformersAdapter requires `torch` and `transformers`.\n"
        "Install them with: pip install torch transformers"
    ) from exc

from .base import ModelAdapter
from srb.types import TokenStep


class LocalTransformersAdapter(ModelAdapter):
    """
    Adapter for local Hugging Face causal language models.

    Example
    -------
    >>> adapter = LocalTransformersAdapter("gpt2")
    >>> steps = adapter.generate_with_traces("Hello world", max_new_tokens=16)
    >>> from srb import analyze_sequence
    >>> result = analyze_sequence(steps)
    """

    def __init__(
        self,
        model: str,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        model:
            Hugging Face model identifier, e.g. "gpt2" or "meta-llama/Llama-3.1-8B".
        device:
            Optional device string, e.g. "cpu", "cuda", or "mps".
            If None, defaults to "cuda" if available, otherwise "mps" if available,
            otherwise "cpu".
        **kwargs:
            Additional generation defaults (e.g., temperature, max_new_tokens)
            stored in the adapter config.
        """
        super().__init__(model=model, **kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        # Ensure we have a pad token for generation if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model)

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate_with_traces(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Sequence[TokenStep]:
        """
        Generate text from the local model and return token-level traces.

        Parameters
        ----------
        prompt:
            Input text to condition the model on.
        **kwargs:
            Generation parameters that override adapter defaults. Common keys:
                - max_new_tokens (int)
                - temperature (float)
                - top_k (int)

        Returns
        -------
        Sequence[TokenStep]
            One TokenStep per generated token.
        """
        cfg: Dict[str, Any] = self.get_effective_config(kwargs)
        max_new_tokens: int = int(cfg.get("max_new_tokens", 64))
        temperature: float = float(cfg.get("temperature", 0.7))
        top_k: Optional[int] = cfg.get("top_k", None)

        # Tokenize prompt
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        prompt_len = input_ids.shape[1]

        # Use generate() to sample tokens, requesting scores for each step.
        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )
        if temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        if top_k is not None:
            gen_kwargs["top_k"] = int(top_k)

        gen_out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

        sequences = gen_out.sequences  # (1, prompt_len + gen_len)
        scores = gen_out.scores        # list of length gen_len, each (1, vocab_size)

        # Compute hidden states for embeddings.
        # Single forward on the full sequence.
        outputs = self.model(sequences, output_hidden_states=True)
        # Use the last layer hidden states as token embeddings.
        hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)

        full_ids = sequences[0]            # (prompt_len + gen_len,)
        gen_ids = full_ids[prompt_len:]    # only newly generated tokens

        steps: List[TokenStep] = []

        vocab_size = scores[0].shape[-1]

        for i, (token_id, step_scores) in enumerate(zip(gen_ids, scores)):
            token_id_int = int(token_id.item())

            # Decode token text (may include spaces / BPE artifacts)
            token_text = self.tokenizer.decode(
                [token_id_int],
                skip_special_tokens=False,
            )

            # Scores -> probs
            logits = step_scores[0]  # (vocab_size,)
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)

            # Chosen token logprob
            logprob = float(log_probs[token_id_int].item())

            probs_list = probs.tolist()
            assert len(probs_list) == vocab_size

            # Embedding: last hidden state at this token position.
            pos = prompt_len + i
            emb_vec = hidden[0, pos, :].tolist()

            steps.append(
                TokenStep(
                    index=i,
                    token=token_text,
                    logprob=logprob,
                    probs=probs_list,
                    embedding=emb_vec,
                )
            )

        return steps