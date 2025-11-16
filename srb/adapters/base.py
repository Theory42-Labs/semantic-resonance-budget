

"""Adapter base classes for integrating models with SRB.

The SRB core library is intentionally model-agnostic. To connect SRB to a
concrete language model backend (e.g., OpenAI, Hugging Face, a local
inference server), you implement a subclass of :class:`ModelAdapter` that
exposes a consistent token-level trace interface.

Adapters are responsible for:

- Running a generation call on a specific model backend.
- Emitting a sequence of :class:`srb.types.TokenStep` objects.
- Populating token text, log probabilities, probability distributions,
  and embeddings as available.

This module defines the abstract base class used by SRB. Concrete adapters
should live alongside this file (e.g., ``openai.py``, ``hf.py``) and may
add backend-specific configuration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Sequence

from srb.types import TokenStep


class ModelAdapter(ABC):
    """Abstract base class for SRB-compatible model adapters.

    Subclasses wrap a specific language model backend and provide a
    standard method for generating token-level traces. The traces are
    returned as a sequence of :class:`TokenStep` objects suitable for
    analysis by :func:`srb.api.analyze_sequence`.
    """

    def __init__(self, model: str, **kwargs: Any) -> None:
        """Initialize the adapter.

        Parameters
        ----------
        model:
            A backend-specific model identifier (e.g., an OpenAI model
            name, a Hugging Face model ID, or a local model handle).
        **kwargs:
            Additional configuration parameters specific to the backend
            (e.g., API keys, endpoints, temperature, max_tokens, etc.).
        """

        self.model = model
        self.config: Dict[str, Any] = dict(kwargs)

    @abstractmethod
    def generate_with_traces(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Sequence[TokenStep]:
        """Generate a completion and return token-level traces.

        Implementations should:

        1. Call the underlying model backend with the given ``prompt``
           and any backend-specific parameters.
        2. Capture token text, logprob (if available), probability
           distribution (full or top-k), and embedding information.
        3. Construct and return a sequence of :class:`TokenStep` objects
           with monotonically increasing ``index`` fields.

        Parameters
        ----------
        prompt:
            The input text to condition the model on.
        **kwargs:
            Additional backend-specific options (e.g., temperature,
            max_tokens, stop sequences, etc.). These may augment or
            override the adapter's default configuration.

        Returns
        -------
        Sequence[TokenStep]
            A sequence of token-level traces suitable for SRB analysis.
        """

        raise NotImplementedError

    def get_effective_config(self, overrides: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        """Return the effective configuration for a generation call.

        This helper merges the adapter's base configuration (provided at
        construction time) with any per-call overrides.
        """

        if overrides is None:
            return dict(self.config)
        merged = dict(self.config)
        merged.update(overrides)
        return merged