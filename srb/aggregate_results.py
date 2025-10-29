"""
SRB Aggregate Results

This script scans a root directory for SRB `trace.csv` files, computes
per-run metrics, aggregates them by semantic bucket (e.g., science,
creative, multilingual), and emits summary tables and comparison plots.

It makes *best-effort* bucket detection by searching any ancestor
folder names for the tokens: {science, creative, multilingual}. If no
match is found, the bucket is labeled "unknown".

Outputs written to the root directory by default:
  - aggregate_per_run.csv
  - aggregate_per_bucket.csv
  - aggregate_global.json
  - figs/
      - bucket_bar_entropy_mean.png
      - bucket_bar_resonance_mean.png
      - bucket_bar_corr.png
      - bucket_bar_lag.png
      - bucket_bar_curvature_rate.png

Note: `aggregate_per_bucket.csv` now includes `stabilized` as a mean fraction.

Usage examples:
  python -m srb.aggregate_results --root reports/longform
  python srb/aggregate_results.py --root reports/longform --figs-dir custom_figs

Dependencies: numpy, pandas, matplotlib
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------
# Config / detection
# ---------------------------------
BUCKET_TOKENS = ("science", "creative", "multilingual")


def detect_bucket_from_path(path: str) -> str:
    """Detect bucket name by scanning path components for known tokens."""
    parts = [p.lower() for p in os.path.normpath(path).split(os.sep)]
    for token in BUCKET_TOKENS:
        for part in parts:
            if token in part:
                return token
    return "unknown"


# ---------------------------------
# I/O helpers
# ---------------------------------

def find_traces_under(root: str) -> List[str]:
    traces: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower() == "trace.csv":
                traces.append(os.path.join(dirpath, fn))
    return sorted(traces)


def read_trace_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic columns
    required = {"step", "entropy", "resonance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}")
    df = df.copy()
    df["step"] = df["step"].astype(int)
    df = df.sort_values("step").reset_index(drop=True)
    # Ensure d_resonance
    if "d_resonance" not in df.columns:
        vals = df["resonance"].values
        df["d_resonance"] = np.diff(vals, prepend=vals[0])
    return df


# ---------------------------------
# Metrics
# ---------------------------------

def xcorr_lag(a: np.ndarray, b: np.ndarray) -> int:
    """Return lag (in steps) maximizing cross-correlation of two 1D series.
    Positive lag means `a` leads `b` (a happens earlier).
    """
    n = min(len(a), len(b))
    a = a[:n] - np.mean(a[:n])
    b = b[:n] - np.mean(b[:n])
    corr = np.correlate(a, b, mode="full")
    lags = np.arange(-n + 1, n)
    return int(lags[np.argmax(corr)])


def curvature_zero_cross_rate(series: np.ndarray) -> float:
    """Approximate rate of sign changes in the second derivative of `series`."""
    d1 = np.gradient(series)
    d2 = np.gradient(d1)
    if len(d2) < 3:
        return 0.0
    signs = np.sign(d2)
    changes = np.sum(signs[1:] * signs[:-1] < 0)
    return changes / max(1, len(d2))


def stabilized_step_estimate(d_resonance: np.ndarray, window: int = 8, eps: float = 0.02) -> Optional[int]:
    """Return first step index where the moving average of |dR| stays below eps for `window` steps.
    Index returned is the *index* within the array; caller can convert to actual step if needed.
    """
    if len(d_resonance) == 0:
        return None
    abs_dr = np.abs(d_resonance)
    # Moving average via convolution
    kernel = np.ones(window) / float(window)
    ma = np.convolve(abs_dr, kernel, mode="same")
    # Find first index where last `window` average is below eps
    # Be conservative: require a full window from the start
    for i in range(window - 1, len(ma)):
        if np.mean(abs_dr[i - window + 1 : i + 1]) < eps:
            return i
    return None


def compute_run_metrics(df: pd.DataFrame, path: str, root: str) -> Dict[str, object]:
    steps = df["step"].values
    H = df["entropy"].values
    R = df["resonance"].values
    dR = df["d_resonance"].values

    # Basic stats
    metrics: Dict[str, object] = {
        "trace_path": path,
        "bucket": detect_bucket_from_path(path),
        "n_steps": int(len(steps)),
        "H_mean": float(np.mean(H)),
        "H_std": float(np.std(H, ddof=0)),
        "H_min": float(np.min(H)),
        "H_max": float(np.max(H)),
        "H_final": float(H[-1]) if len(H) else np.nan,
        "R_mean": float(np.mean(R)),
        "R_std": float(np.std(R, ddof=0)),
        "R_min": float(np.min(R)),
        "R_max": float(np.max(R)),
        "R_final": float(R[-1]) if len(R) else np.nan,
        "HR_corr": float(pd.Series(H).corr(pd.Series(R))) if len(H) > 1 else np.nan,
        "neg_dR_count": int(np.sum(dR < 0)),
        "abs_dR_mean": float(np.mean(np.abs(dR))) if len(dR) else np.nan,
    }

    # Lag (entropy -> resonance)
    try:
        metrics["lag_H_to_R_steps"] = int(xcorr_lag(H, R))
    except Exception:
        metrics["lag_H_to_R_steps"] = np.nan

    # Entropy slope (per step, least squares)
    try:
        x = steps.astype(float)
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, H, rcond=None)[0]
        metrics["H_slope_per_step"] = float(slope)
    except Exception:
        metrics["H_slope_per_step"] = np.nan

    # Curvature zero-cross rate
    metrics["curvature_zero_rate"] = float(curvature_zero_cross_rate(H))

    # Stabilized index estimate from |dR|
    stab_idx = stabilized_step_estimate(dR, window=8, eps=0.02)
    metrics["stabilized"] = bool(stab_idx is not None)
    # If not stabilized, fall back to the final index (len(dR) - 1) to indicate "ran full length without stabilizing"
    if stab_idx is not None:
        metrics["stabilized_index"] = int(stab_idx)
    else:
        metrics["stabilized_index"] = int(max(0, len(dR) - 1))

    # Derive run_id and run_root
    run_root = os.path.dirname(path)
    run_id = os.path.basename(run_root)
    metrics["run_id"] = run_id
    metrics["run_root"] = run_root

    return metrics


# ---------------------------------
# Plotting
# ---------------------------------

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _bar(ax, xlabels: List[str], values: List[float], ylabel: str, title: str):
    ax.bar(range(len(xlabels)), values)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def make_bucket_bar_figures(df_bucket: pd.DataFrame, figs_dir: str) -> None:
    cats = df_bucket.index.tolist()

    def save_one(col: str, ylabel: str, title: str, fname: str):
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        vals = df_bucket[col].values.tolist()
        _bar(ax, cats, vals, ylabel, title)
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, fname), dpi=180, bbox_inches="tight")
        plt.close(fig)

    save_one("H_mean", "entropy (mean)", "Mean Entropy by Bucket", "bucket_bar_entropy_mean.png")
    save_one("R_mean", "resonance (mean)", "Mean Resonance by Bucket", "bucket_bar_resonance_mean.png")
    if "HR_corr" in df_bucket.columns:
        save_one("HR_corr", "corr(H,R)", "Entropy–Resonance Correlation by Bucket", "bucket_bar_corr.png")
    if "lag_H_to_R_steps" in df_bucket.columns:
        save_one("lag_H_to_R_steps", "steps", "Semantic Lag (H→R) by Bucket", "bucket_bar_lag.png")
    if "curvature_zero_rate" in df_bucket.columns:
        save_one("curvature_zero_rate", "rate", "Entropy Curvature Zero-Cross Rate by Bucket", "bucket_bar_curvature_rate.png")


# ---------------------------------
# Main
# ---------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate SRB trace metrics across runs and buckets")
    parser.add_argument("--root", type=str, required=True, help="Root directory to scan recursively for trace.csv files")
    parser.add_argument("--figs-dir", type=str, default=None, help="Directory for aggregate figures (default: <root>/figs)")
    parser.add_argument("--out-prefix", type=str, default=None, help="File prefix for CSV/JSON (default: <root>/aggregate_*)")
    args = parser.parse_args(argv)

    root = os.path.abspath(args.root)
    figs_dir = os.path.abspath(args.figs_dir) if args.figs_dir else os.path.join(root, "figs")
    out_prefix = args.out_prefix or os.path.join(root, "aggregate")

    traces = find_traces_under(root)
    if not traces:
        print(f"[SRB Aggregate] No trace.csv files found under: {root}")
        return 1

    per_run_rows: List[Dict[str, object]] = []
    for p in traces:
        try:
            df = read_trace_csv(p)
            row = compute_run_metrics(df, p, root)
            per_run_rows.append(row)
        except Exception:
            print(f"[SRB Aggregate] Failed to process: {p}")
            traceback.print_exc()

    if not per_run_rows:
        print("[SRB Aggregate] No runs processed successfully.")
        return 2

    df_runs = pd.DataFrame(per_run_rows)

    # Per-bucket aggregation: mean of run-level metrics
    numeric_cols = [c for c in df_runs.columns if c not in {"trace_path", "bucket", "run_id", "run_root"}]
    df_bucket = df_runs.groupby("bucket")[numeric_cols].mean().sort_index()

    # Global summary: mean of bucket means
    global_summary = df_bucket.mean(numeric_only=True).to_dict()

    # Write outputs
    out_runs_csv = f"{out_prefix}_per_run.csv"
    out_bucket_csv = f"{out_prefix}_per_bucket.csv"
    out_global_json = f"{out_prefix}_global.json"

    df_runs.to_csv(out_runs_csv, index=False)
    df_bucket.to_csv(out_bucket_csv)
    with open(out_global_json, "w") as f:
        json.dump(global_summary, f, indent=2)

    # Figures
    _ensure_dir(figs_dir)
    make_bucket_bar_figures(df_bucket, figs_dir)

    print("[SRB Aggregate] Wrote:")
    print("  ", out_runs_csv)
    print("  ", out_bucket_csv)
    print("  ", out_global_json)
    print("  Figures →", figs_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
