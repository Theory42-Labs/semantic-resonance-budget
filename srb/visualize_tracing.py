"""
SRB Trace Visualizer

Generates three figures for every `trace.csv` it finds under a given root
(or for a single file):
  1) Entropy & Resonance over Time (dual-axis overlay)
  2) Phase-Space Flow (quiver of dR/dt vs dH/dt in R–H space)
  3) Entropy Curvature Map (second derivative of entropy vs step)

Each image is saved alongside the source `trace.csv`.

Usage examples:
  python -m srb.visualize_tracing --root reports/longform
  python -m srb.visualize_tracing --file reports/longform/20251029_1237/run_00/trace.csv

Dependencies: numpy, pandas, matplotlib (no seaborn required)
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Data loading & validation
# ----------------------------
REQUIRED_COLS = {"step", "entropy", "resonance"}
OPTIONAL_COLS = {"d_resonance", "entropy_plateau", "soft_stop", "reflection_triggered"}


def _read_trace_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}")
    # Ensure correct dtypes & ordering
    df = df.copy()
    df["step"] = df["step"].astype(int)
    df = df.sort_values("step").reset_index(drop=True)
    # If d_resonance absent, compute finite difference
    if "d_resonance" not in df.columns:
        dr = np.diff(df["resonance"].values, prepend=df["resonance"].values[0])
        df["d_resonance"] = dr
    # Coerce optional boolean-ish columns
    for col in (OPTIONAL_COLS & set(df.columns)):
        # Interpret any nonzero/nonempty as True
        if df[col].dtype != bool:
            df[col] = df[col].astype(float).fillna(0.0) != 0.0
    return df


# ----------------------------
# Plot helpers
# ----------------------------
@dataclass
class FigurePaths:
    time_overlay: str
    phase_quiver: str
    entropy_curvature: str


def _ensure_outdir_for(path_to_csv: str) -> str:
    outdir = os.path.dirname(os.path.abspath(path_to_csv))
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)
    return outdir


def _safe_savefig(out_path: str) -> None:
    # Keep defaults (no custom styles/colors) per project guidance
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_entropy_resonance_time(df: pd.DataFrame, csv_path: str, ma_window: int = 7) -> str:
    outdir = _ensure_outdir_for(csv_path)
    out_path = os.path.join(outdir, "entropy_resonance_over_time.png")

    steps = df["step"].values
    H = df["entropy"].values
    R = df["resonance"].values

    # Rolling means for visual smoothing (optional)
    if ma_window and ma_window > 1:
        H_ma = pd.Series(H).rolling(ma_window, min_periods=1).mean().values
        R_ma = pd.Series(R).rolling(ma_window, min_periods=1).mean().values
    else:
        H_ma, R_ma = None, None

    fig, ax1 = plt.subplots(figsize=(9, 4.8))

    # Distinct colors for clarity
    color_entropy = "tab:blue"
    color_resonance = "tab:orange"

    # Raw series
    l1_raw, = ax1.plot(steps, H, color=color_entropy, alpha=0.35, label="entropy (raw)")
    ax1.set_xlabel("step")
    ax1.set_ylabel("entropy", color=color_entropy)
    ax1.tick_params(axis="y", labelcolor=color_entropy)

    ax2 = ax1.twinx()
    l2_raw, = ax2.plot(steps, R, color=color_resonance, alpha=0.35, label="resonance (raw)")
    ax2.set_ylabel("resonance", color=color_resonance)
    ax2.tick_params(axis="y", labelcolor=color_resonance)

    legend_items = [l1_raw, l2_raw]

    # Smoothed overlays (moving average)
    if H_ma is not None and R_ma is not None:
        l1_ma, = ax1.plot(steps, H_ma, color="tab:green", linewidth=2.0, label=f"entropy (MA{ma_window})")
        l2_ma, = ax2.plot(steps, R_ma, color="tab:red", linewidth=2.0, label=f"resonance (MA{ma_window})")
        legend_items.extend([l1_ma, l2_ma])

    # Event markers if available
    if "entropy_plateau" in df.columns and df["entropy_plateau"].any():
        idx = np.where(df["entropy_plateau"].values)[0]
        plateau_scatter = ax1.scatter(steps[idx], H[idx], marker="o", color="purple", label="entropy_plateau")
        legend_items.append(plateau_scatter)
    if "soft_stop" in df.columns and df["soft_stop"].any():
        idx = np.where(df["soft_stop"].values)[0]
        soft_scatter = ax1.scatter(steps[idx], H[idx], marker="x", color="red", label="soft_stop")
        legend_items.append(soft_scatter)
    if "reflection_triggered" in df.columns and df["reflection_triggered"].any():
        idx = np.where(df["reflection_triggered"].values)[0]
        refl_scatter = ax2.scatter(steps[idx], R[idx], marker="s", color="green", label="reflection_triggered")
        legend_items.append(refl_scatter)

    fig.legend(handles=legend_items, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    _safe_savefig(out_path)
    return out_path


def plot_phase_space_quiver(df: pd.DataFrame, csv_path: str) -> str:
    outdir = _ensure_outdir_for(csv_path)
    out_path = os.path.join(outdir, "phase_space_quiver.png")

    H = df["entropy"].values
    R = df["resonance"].values

    # First derivatives w.r.t. step
    dH = np.gradient(H)
    dR = np.gradient(R)

    # Downsample for clarity on long runs
    n = len(df)
    stride = max(1, n // 80)
    idx = np.arange(0, n, stride)

    plt.figure(figsize=(7.8, 6.8))
    plt.plot(H, R)
    plt.quiver(H[idx], R[idx], dH[idx], dR[idx], angles='xy', scale_units='xy', scale=1.0, width=0.0035)
    plt.xlabel("entropy (H)")
    plt.ylabel("resonance (R)")
    plt.title("Phase-Space Flow (ΔR vs ΔH)")
    _safe_savefig(out_path)
    return out_path


def plot_entropy_curvature(df: pd.DataFrame, csv_path: str) -> str:
    outdir = _ensure_outdir_for(csv_path)
    out_path = os.path.join(outdir, "entropy_curvature.png")

    steps = df["step"].values
    H = df["entropy"].values

    # First and second derivatives
    dH_dt = np.gradient(H)
    d2H_dt2 = np.gradient(dH_dt)

    plt.figure(figsize=(9, 4.8))
    plt.plot(steps, d2H_dt2, label="d²H/dt²")

    # Mark sign changes (potential inflection points)
    sign = np.sign(d2H_dt2)
    sign_change_idx = np.where(np.diff(sign) != 0)[0]
    if len(sign_change_idx):
        plt.scatter(steps[sign_change_idx], d2H_dt2[sign_change_idx])

    plt.xlabel("step")
    plt.ylabel("entropy curvature (d²H/dt²)")
    plt.title("Entropy Curvature over Time")
    plt.legend()
    _safe_savefig(out_path)
    return out_path


# ----------------------------
# Orchestration
# ----------------------------

# Default moving-average window for smoothing in the time overlay plot
PROCESS_MA_WINDOW: int = 7

def process_trace_csv(csv_path: str) -> Optional[FigurePaths]:
    try:
        df = _read_trace_csv(csv_path)
        time_img = plot_entropy_resonance_time(df, csv_path, ma_window=PROCESS_MA_WINDOW)
        phase_img = plot_phase_space_quiver(df, csv_path)
        curv_img = plot_entropy_curvature(df, csv_path)
        print(f"[SRB Viz] Saved: \n  {time_img}\n  {phase_img}\n  {curv_img}")
        return FigurePaths(time_img, phase_img, curv_img)
    except Exception:
        print(f"[SRB Viz] Failed to process {csv_path}")
        traceback.print_exc()
        return None


def find_traces_under(root: str) -> List[str]:
    traces: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower() == "trace.csv":
                traces.append(os.path.join(dirpath, fn))
    return sorted(traces)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SRB trace visualizer")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--root", type=str, help="Root directory to recursively scan for trace.csv files")
    g.add_argument("--file", type=str, help="Path to a single trace.csv file")
    parser.add_argument("--ma-window", type=int, default=7, help="Moving-average window for smoothing the time overlay (set 1 to disable)")
    args = parser.parse_args(argv)

    global PROCESS_MA_WINDOW
    PROCESS_MA_WINDOW = max(1, int(args.ma_window))

    targets: List[str]
    if args.file:
        targets = [args.file]
    else:
        targets = find_traces_under(args.root)
        if not targets:
            print(f"[SRB Viz] No trace.csv files found under: {args.root}")
            return 1

    ok, fail = 0, 0
    for csv_path in targets:
        res = process_trace_csv(csv_path)
        if res is None:
            fail += 1
        else:
            ok += 1
    print(f"[SRB Viz] Done. Success: {ok}, Failed: {fail}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())