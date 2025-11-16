

"""
SRB-CST: Conceptual State Test

This module implements a small benchmark harness for running the
Semantic Resonance Budget – Conceptual State Test (SRB-CST) against
any SRB-compatible adapter.

The test probes four conceptual conditions:

1. Unknown concept
2. Known scientific concept
3. Fictional hybrid concept
4. User-defined (in-context) concept

For each condition it:
- generates a short completion with token-level traces,
- computes SRB metrics,
- aggregates key statistics,
- optionally plots velocity / resonance curves.

See `docs/Conceptual_State_Test.md` and `docs/Cognitive_Signatures.md`
for the conceptual background and interpretation guidance.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

from srb import analyze_sequence
from srb.resonance import build_resonance_from_sequence, estimate_collapse_step
from srb.viz import plot_metric_series, plot_resonance_series


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ConditionConfig:
    """Configuration for a single conceptual-state condition."""

    key: str
    label: str
    prompt: str
    description: str


@dataclass
class ConditionResult:
    """Raw SRB result for a single condition."""

    config: ConditionConfig
    steps: Any
    aggregates: Dict[str, float]
    resonance_steps: Any
    collapse_idx: Optional[int]


@dataclass
class ConceptualStateTestResult:
    """
    Container for all SRB-CST results.

    Attributes
    ----------
    conditions:
        List of ConditionResult objects in evaluation order.
    """

    conditions: List[ConditionResult]

    # ---- tabular views -------------------------------------------------

    def to_rows(self) -> List[Dict[str, Any]]:
        """Return results as a list of dict rows (for DataFrame or printing)."""
        rows: List[Dict[str, Any]] = []
        for cond in self.conditions:
            row: Dict[str, Any] = {
                "key": cond.config.key,
                "label": cond.config.label,
                "prompt": cond.config.prompt,
                "collapse_idx": cond.collapse_idx,
            }
            row.update(cond.aggregates)
            rows.append(row)
        return rows

    @property
    def table(self) -> Any:
        """
        Return a pandas.DataFrame view of the results if pandas is available.

        If pandas is not installed, this returns the list-of-dicts representation
        from `to_rows()`.
        """
        rows = self.to_rows()
        if pd is None:
            return rows
        return pd.DataFrame(rows)

    # ---- pretty-printing -----------------------------------------------

    def pprint(self) -> None:
        """Print a human-friendly summary of the benchmark results."""
        if pd is not None:
            print(self.table.to_string(index=False))
        else:
            for row in self.to_rows():
                print("-" * 80)
                print(f"{row['label']} ({row['key']})")
                for k, v in row.items():
                    if k in {"key", "label", "prompt"}:
                        continue
                    print(f"  {k}: {v}")
            print("-" * 80)

    # ---- plotting helpers ----------------------------------------------

    def plot_condition(
        self,
        key: str,
        metric: str = "semantic_velocity",
        show_resonance: bool = True,
    ) -> None:
        """
        Plot SRB curves for a single condition.

        Parameters
        ----------
        key:
            Condition key, e.g. "unknown", "known", "fictional", "defined".
        metric:
            Name of the metric in SRBSequenceMetrics.metrics to plot on the
            first chart. Default: "semantic_velocity".
        show_resonance:
            If True, also plot resonance / coherence / normalized entropy
            curves for the condition.
        """
        cond = self._get_condition(key)
        result_like = type("ResultLike", (), {})()
        setattr(result_like, "metrics", {})  # placeholder, will be filled below

        # We reconstruct a shallow result-like object because plot_metric_series
        # expects an SRBSequenceMetrics-like structure.
        analyzed = analyze_sequence(cond.steps)
        setattr(result_like, "metrics", analyzed.metrics)

        plot_metric_series(
            analyzed,
            metric_key=metric,
            title=f"{cond.config.label} – {metric}",
            show=True,
        )

        if show_resonance:
            plot_resonance_series(
                cond.resonance_steps,
                collapse_idx=cond.collapse_idx,
                title=f"{cond.config.label} – resonance / coherence / entropy_norm",
                show=True,
            )

    def _get_condition(self, key: str) -> ConditionResult:
        for cond in self.conditions:
            if cond.config.key == key:
                return cond
        raise KeyError(f"No condition found with key={key!r}")


# ---------------------------------------------------------------------------
# Default condition set
# ---------------------------------------------------------------------------


def default_conditions() -> List[ConditionConfig]:
    """
    Return the default SRB-CST condition set.

    These correspond to the four canonical conceptual states:

    - unknown concept
    - known scientific concept
    - fictional hybrid concept
    - user-defined (in-context) concept
    """
    return [
        ConditionConfig(
            key="unknown",
            label="Unknown concept: SRB",
            prompt="Explain Semantic Resonance Budget in one paragraph.",
            description=(
                "A deliberately unknown concept. The model has no prior grounding "
                "for 'Semantic Resonance Budget' and will typically fall back to "
                "generic filler or unrelated explanations."
            ),
        ),
        ConditionConfig(
            key="known",
            label="Known concept: entropy regularization",
            prompt=(
                "Explain entropy regularization in reinforcement learning in one "
                "paragraph. Include both intuitive and technical aspects."
            ),
            description=(
                "A well-known concept from machine learning literature that most "
                "modern LLMs should have encountered during pretraining."
            ),
        ),
        ConditionConfig(
            key="fictional",
            label="Fictional concept: arcane thermodynamics",
            prompt=(
                "Explain the mechanics of quantized mana resonance in arcane "
                "thermodynamics. Describe how energy flows, what constraints "
                "apply, and how stability is achieved."
            ),
            description=(
                "A hybrid of fantasy and physics terminology. The model will often "
                "respond with confident but ungrounded 'science-flavored' prose."
            ),
        ),
        ConditionConfig(
            key="defined",
            label="Defined concept: SRB with provided definition",
            prompt=(
                "Semantic Resonance Budget is an analysis framework that looks at "
                "each token generated by a language model and measures how its "
                "meaning moves over time. It combines several signals: "
                "- semantic velocity (how far embeddings move between tokens), "
                "- entropy (how uncertain the model is), "
                "- surprisal dynamics, and "
                "- divergence from a normative semantic baseline.\n\n"
                "By combining these, SRB can show when a model is reasoning in a "
                "stable, coherent way versus when it is guessing, drifting, or "
                "collapsing.\n\n"
                "In your own words, explain Semantic Resonance Budget in one "
                "paragraph, as if you are teaching it to a curious engineer."
            ),
            description=(
                "A user-defined concept whose meaning is introduced directly in "
                "the prompt, testing the model's ability to integrate and reuse a "
                "new conceptual scaffold in-context."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------


def run_condition(adapter: Any, config: ConditionConfig, verbose: bool = True) -> ConditionResult:
    """
    Run SRB analysis for a single conceptual condition.

    Parameters
    ----------
    adapter:
        Any SRB-compatible adapter exposing `generate_with_traces(prompt: str)`.
    config:
        ConditionConfig describing the prompt and label.
    verbose:
        If True, prints the prompt, model output, and aggregate metrics.

    Returns
    -------
    ConditionResult
    """
    if verbose:
        print("=" * 80)
        print(f"EXPERIMENT: {config.label}")
        print("-" * 80)
        print("Prompt:")
        print("  " + config.prompt.replace("\n", "\n  "))
        print("-" * 80)

    steps = adapter.generate_with_traces(config.prompt)
    if verbose:
        print(f"Generated {len(steps)} tokens\n")

        generated_text = "".join(step.token for step in steps)
        print("Model Output (approximate reconstruction):")
        # Basic wrapping without importing textwrap (to keep deps minimal here)
        for line in generated_text.split("\n"):
            print("  " + line)
        print()

    analyzed = analyze_sequence(steps)
    aggregates = {k: float(v) for k, v in analyzed.aggregates.items()}

    resonance_steps = build_resonance_from_sequence(analyzed)
    collapse_idx = estimate_collapse_step(
        [s.resonance for s in resonance_steps],
        window=5,
        eps=1e-3,
    )

    if verbose:
        print("SRB Aggregates:")
        for k, v in aggregates.items():
            print(f"  {k}: {v:.6f}")
        print()
        print(f"Resonance collapse index: {collapse_idx}")
        print()

    return ConditionResult(
        config=config,
        steps=steps,
        aggregates=aggregates,
        resonance_steps=resonance_steps,
        collapse_idx=collapse_idx,
    )


def run_conceptual_state_test(
    adapter: Any,
    conditions: Optional[List[ConditionConfig]] = None,
    verbose: bool = True,
    plot: bool = False,
) -> ConceptualStateTestResult:
    """
    Run the full SRB Conceptual State Test (SRB-CST) for a given adapter.

    Parameters
    ----------
    adapter:
        Any SRB-compatible adapter exposing `generate_with_traces(prompt: str)`.
    conditions:
        Optional custom condition set. If None, uses `default_conditions()`.
    verbose:
        If True, prints details for each condition and a summary table.
    plot:
        If True, generates velocity and resonance plots for all conditions.

    Returns
    -------
    ConceptualStateTestResult
    """
    if conditions is None:
        conditions = default_conditions()

    results: List[ConditionResult] = []
    for cfg in conditions:
        results.append(run_condition(adapter, cfg, verbose=verbose))

    test_result = ConceptualStateTestResult(conditions=results)

    if verbose:
        print("=" * 80)
        print("SRB-CST SUMMARY")
        print("=" * 80)
        test_result.pprint()

    if plot:
        # Plot semantic velocity and resonance for each condition
        from srb.kinematics import SEMANTIC_VELOCITY_KEY  # type: ignore

        for cond in results:
            analyzed = analyze_sequence(cond.steps)
            plot_metric_series(
                analyzed,
                metric_key=SEMANTIC_VELOCITY_KEY,
                title=f"{cond.config.label} – semantic velocity",
                show=True,
            )
            plot_resonance_series(
                cond.resonance_steps,
                collapse_idx=cond.collapse_idx,
                title=f"{cond.config.label} – resonance / coherence / entropy_norm",
                show=True,
            )

    return test_result


# ---------------------------------------------------------------------------
# CLI entry point (optional helper)
# ---------------------------------------------------------------------------


def _cli() -> None:  # pragma: no cover
    """
    Simple CLI helper to run SRB-CST with a local transformers adapter.

    Example:
        python -m benchmarks.conceptual_state_test
    """
    from srb.adapters import LocalTransformersAdapter

    model_name = "microsoft/phi-2"
    print(f"Initializing LocalTransformersAdapter with model={model_name!r}...")
    adapter = LocalTransformersAdapter(
        model_name,
        max_new_tokens=32,
        temperature=0.7,
    )

    result = run_conceptual_state_test(adapter, verbose=True, plot=False)
    if pd is not None:
        print("\nDataFrame view:")
        print(result.table)


if __name__ == "__main__":  # pragma: no cover
    _cli()