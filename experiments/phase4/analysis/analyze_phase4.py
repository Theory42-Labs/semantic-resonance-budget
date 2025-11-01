#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4 Analysis:
- Finds the latest run folder under experiments/phase4/reports unless --run_dir is provided
- Loads phase4_summary.csv
- Aggregates per-intervention means
- Computes simple effect sizes vs baseline (Cohen's d using pooled std)
- Writes analysis_summary.csv and a few PNG plots into the run folder
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

METRICS = [
    "transfer_entropy",
    "cross_entropy_nll",
    "nsm_coherence",
    "verifier_similarity",
    "ref_similarity",
    "traj_drift",
    "betti0",
    "betti1",
]

def find_latest_run(base: Path) -> Path:
    runs = sorted([p for p in base.glob("run_*") if p.is_dir()], key=lambda p: p.name)
    if not runs:
        raise FileNotFoundError(f"No run_* folders found in {base}")
    return runs[-1]

def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    # x = baseline, y = other
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    nx, ny = len(x), len(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    sp = np.sqrt(((nx-1)*sx**2 + (ny-1)*sy**2) / (nx+ny-2))
    if sp == 0:
        return np.nan
    return (np.mean(y) - np.mean(x)) / sp

def main():
    ap = argparse.ArgumentParser(description="Analyze Phase 4 results")
    ap.add_argument("--run_dir", type=str, help="Path to a specific run_* directory")
    ap.add_argument("--reports_root", type=str, default="experiments/phase4/reports",
                    help="Root reports dir to search for latest run if --run_dir not provided")
    args = ap.parse_args()

    reports_root = Path(args.reports_root)
    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run(reports_root)

    csv_path = run_dir / "phase4_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)

    # Basic sanity
    missing = [m for m in METRICS if m not in df.columns]
    if missing:
        print(f"Warning: missing metrics in CSV: {missing}")

    # Per-intervention means
    agg = (
        df.groupby(["bucket", "intervention"])[[m for m in METRICS if m in df.columns]]
        .mean()
        .reset_index()
        .sort_values(["bucket", "intervention"])
    )

    # Effect sizes vs baseline, per-bucket & metric
    effects = []
    for bucket, sub in df.groupby("bucket"):
        base = sub[sub["intervention"] == "baseline"]
        if base.empty:
            continue
        for inter, sub2 in sub.groupby("intervention"):
            if inter == "baseline":
                continue
            for m in METRICS:
                if m not in sub.columns:
                    continue
                d = cohen_d(base[m].to_numpy(), sub2[m].to_numpy())
                effects.append({"bucket": bucket, "intervention": inter, "metric": m, "cohens_d": d})

    eff_df = pd.DataFrame(effects)

    # Save numeric summaries
    out_summary = run_dir / "analysis_summary.csv"
    out_effects = run_dir / "analysis_effect_sizes.csv"
    agg.to_csv(out_summary, index=False)
    eff_df.to_csv(out_effects, index=False)

    # Quick plots: per-intervention means for key metrics (collapsed over buckets)
    mean_overall = (
        df.groupby("intervention")[[m for m in METRICS if m in df.columns]]
        .mean()
        .reset_index()
        .sort_values("intervention")
    )

    # Plot helper
    def barplot(metric: str, fname: str):
        if metric not in mean_overall.columns:
            return
        plt.figure(figsize=(9, 4.5))
        xs = mean_overall["intervention"].tolist()
        ys = mean_overall[metric].tolist()
        plt.bar(xs, ys)
        plt.title(f"Phase 4 — mean {metric} by intervention")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(run_dir / fname, dpi=150)
        plt.close()

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    barplot("ref_similarity", f"plot_ref_similarity_{ts}.png")
    barplot("traj_drift", f"plot_traj_drift_{ts}.png")
    barplot("nsm_coherence", f"plot_nsm_coherence_{ts}.png")
    barplot("verifier_similarity", f"plot_verifier_similarity_{ts}.png")
    barplot("betti1", f"plot_betti1_{ts}.png")

    # Also produce a bucket-stratified table for quick inspection
    pivot = (
        df.pivot_table(
            index=["bucket", "intervention"],
            values=[m for m in METRICS if m in df.columns],
            aggfunc="mean",
        )
        .reset_index()
        .sort_values(["bucket", "intervention"])
    )
    pivot.to_csv(run_dir / f"analysis_bucket_means_{ts}.csv", index=False)

    print(f"✅ Wrote:\n- {out_summary}\n- {out_effects}\n- plots & bucket means into {run_dir}")

if __name__ == "__main__":
    main()