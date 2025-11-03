"""
Phase 5 — Analysis Toolkit (local quick-run edition)

This script ingests the per-step temporal records from Phase 5 runs and
writes analysis artifacts **inside the same timestamped run directory** by default.

Typical layout (created by run_phase5.py):
  experiments/phase5/output/<run_id_YYYYMMDD_HHMMSS>/
    ├─ phase5_records.jsonl
    ├─ config.yml (run config snapshot; optional)
    └─ analysis_phase5/  (created by this script)

Usage examples:
  # Auto-detect latest run dir under experiments/phase5/output
  python analyze_phase5.py

  # Or analyze a specific run directory
  python analyze_phase5.py --run-dir experiments/phase5/output/srb_local_run_20251103_070845

  # Or write artifacts to a custom output directory
  python analyze_phase5.py --run-dir <run_dir> --out <custom_out>
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: UMAP + ripser; if missing we skip gracefully
try:
    import umap
except Exception:
    umap = None

try:
    from ripser import ripser
    from persim import plot_diagrams
except Exception:
    ripser = None
    plot_diagrams = None

import argparse
import os
from datetime import datetime
try:
    import yaml
except Exception:
    yaml = None
from sklearn.decomposition import PCA


# ----------------------------
# Helpers for run discovery & config
# ----------------------------

def find_latest_run(base: Path) -> Optional[Path]:
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir() and (p / "phase5_records.jsonl").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_cfg_yaml(run_dir: Path) -> dict:
    cfgp = run_dir / "config.yml"
    if yaml is None or not cfgp.exists():
        return {}
    try:
        return yaml.safe_load(cfgp.read_text()) or {}
    except Exception:
        return {}

# Numerical stability for TDA on small, tight manifolds

def reduce_for_tda(X: np.ndarray) -> np.ndarray:
    n, d = X.shape
    k = max(2, min(32, d, n - 1))
    if k < d:
        Xr = PCA(n_components=k, random_state=42).fit_transform(X)
    else:
        Xr = X.copy()
    # tiny jitter to avoid singularities
    if n <= 64:
        Xr = Xr + 1e-6 * np.random.default_rng(42).standard_normal(Xr.shape)
    return Xr


# ----------------------------
# Data loading / preparation
# ----------------------------

@dataclass
class StepRecord:
    run_id: str
    prompt_id: str
    prompt_text: str
    t: int
    token: str
    token_id: int
    logprob: float
    surprisal: float
    cum_text: str           # cumulative text up to step t
    cos_to_prev: Optional[float] = None
    cos_to_prompt: Optional[float] = None
    bucket: Optional[str] = None
    model_name: Optional[str] = None


def load_records(jsonl_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Expect a "type" field == "phase5_step"
            if obj.get("type") != "phase5_step":
                continue
            rows.append(obj)
    if not rows:
        raise FileNotFoundError(f"No phase5_step records found in {jsonl_path}")
    df = pd.DataFrame(rows)
    # enforce types / default columns
    for col in ["run_id", "prompt_id", "prompt_text", "cum_text", "bucket", "model_name"]:
        if col not in df.columns:
            df[col] = ""
    for col in ["t", "token_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
        else:
            df[col] = -1
    for col in ["logprob", "surprisal", "cos_to_prev", "cos_to_prompt"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    if "token" not in df.columns:
        df["token"] = ""
    return df


# ----------------------------
# Helpers for semantic kinematics and prompt half-life
# ----------------------------

def prompt_half_life(series: pd.Series, frac: float = 0.95) -> Optional[int]:
    """Return first token index where cos_to_prompt ≤ frac * initial (first finite) value."""
    vals = pd.to_numeric(series, errors="coerce").values
    finite = np.isfinite(vals)
    if not finite.any():
        return None
    c0 = vals[finite][0]
    if not np.isfinite(c0):
        return None
    thresh = c0 * frac
    for i, v in enumerate(vals):
        if np.isfinite(v) and v <= thresh:
            return i
    return None


def compute_semantic_kinematics(Z: np.ndarray) -> dict:
    """
    Given a 2D trajectory Z (n,2), compute per-step semantic speed and curvature.
    speed[t] = ||Z[t] - Z[t-1]|| for t >= 1
    curvature[t] for t >= 2 is the turning angle (radians) between successive step vectors.
    """
    n = len(Z)
    if n < 3:
        return {"speed": np.array([]), "curvature": np.array([])}
    diffs = Z[1:] - Z[:-1]
    speed = np.linalg.norm(diffs, axis=1)
    v_prev = diffs[:-1]
    v_curr = diffs[1:]
    denom = (np.linalg.norm(v_prev, axis=1) * np.linalg.norm(v_curr, axis=1)) + 1e-12
    dots = (v_prev * v_curr).sum(axis=1) / denom
    dots = np.clip(dots, -1.0, 1.0)
    curvature = np.arccos(dots)
    return {"speed": speed, "curvature": curvature}


# ----------------------------
# Plots
# ----------------------------

def ensure_outdir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    return base


def plot_temporal_traces(df: pd.DataFrame, outdir: Path, label: str) -> Path:
    """
    Plot surprisal and cosine coherence over time for a single (run_id, prompt_id) series.
    """
    series = df.sort_values("t")
    t = series["t"].values
    s = series["surprisal"].values
    cprev = series["cos_to_prev"].values
    cprompt = series["cos_to_prompt"].values

    # Surprisal trace
    plt.figure(figsize=(8, 4.5))
    plt.plot(t, s, marker="o")
    plt.title(f"Surprisal over time — {label}")
    plt.xlabel("t (token step)")
    plt.ylabel("surprisal (−log p)")
    sp = outdir / f"{label}_trace_surprisal.png"
    plt.tight_layout()
    plt.savefig(sp, dpi=180)
    plt.close()

    # Cosine to previous
    plt.figure(figsize=(8, 4.5))
    plt.plot(t, cprev, marker="o")
    plt.title(f"Coherence to previous over time — {label}")
    plt.xlabel("t (token step)")
    plt.ylabel("cosine(cum_t, cum_{t-1})")
    cp = outdir / f"{label}_trace_cos_prev.png"
    plt.tight_layout()
    plt.savefig(cp, dpi=180)
    plt.close()

    # Cosine to prompt
    plt.figure(figsize=(8, 4.5))
    plt.plot(t, cprompt, marker="o")
    plt.title(f"Coherence to prompt over time — {label}")
    plt.xlabel("t (token step)")
    plt.ylabel("cosine(cum_t, prompt)")
    cpp = outdir / f"{label}_trace_cos_prompt.png"
    plt.tight_layout()
    plt.savefig(cpp, dpi=180)
    plt.close()

    return outdir


def embed_texts_sbert(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Lightweight sentence embedding for trajectory plotting.
    """
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(model_name)
    X = enc.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
    return X


def plot_umap_trajectory(df: pd.DataFrame, outdir: Path, label: str) -> Optional[tuple[Path, np.ndarray]]:
    if umap is None:
        print("[analyze_phase5] UMAP not available; skipping trajectory plot.")
        return None

    series = df.sort_values("t")
    texts = series["cum_text"].tolist()
    try:
        X = embed_texts_sbert(texts)
    except Exception as e:
        print(f"[analyze_phase5] Embedding failed: {e}")
        return None

    # Reduce dimensionality for numerical stability
    Xr = reduce_for_tda(X)
    reducer = umap.UMAP(
        n_neighbors=min(10, max(2, len(Xr) - 2)),
        n_components=2,
        random_state=42,
        metric="euclidean",
    )
    Z = reducer.fit_transform(Xr)

    plt.figure(figsize=(6.5, 6.0))
    plt.plot(Z[:, 0], Z[:, 1], marker="o", linewidth=1.0)
    for i, (x, y) in enumerate(Z):
        if i % max(1, len(Z)//12) == 0:
            plt.text(x, y, str(i), fontsize=8)
    plt.title(f"UMAP trajectory — {label}")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    figp = outdir / f"{label}_umap_trajectory.png"
    plt.tight_layout()
    plt.savefig(figp, dpi=180)
    plt.close()
    return figp, Z


def plot_semantic_velocity(speed: np.ndarray, outdir: Path, label: str) -> Optional[Path]:
    if speed.size == 0:
        return None
    t = np.arange(1, speed.size + 1)
    plt.figure(figsize=(8, 4.5))
    plt.plot(t, speed, marker="o")
    plt.title(f"Semantic velocity — {label}")
    plt.xlabel("t (token step)")
    plt.ylabel("UMAP Δdistance per step")
    p = outdir / f"{label}_semantic_velocity.png"
    plt.tight_layout()
    plt.savefig(p, dpi=180)
    plt.close()
    return p


def plot_semantic_curvature(curv: np.ndarray, outdir: Path, label: str) -> Optional[Path]:
    if curv.size == 0:
        return None
    t = np.arange(2, curv.size + 2)
    plt.figure(figsize=(8, 4.5))
    plt.plot(t, curv, marker="o")
    plt.title(f"Semantic curvature — {label}")
    plt.xlabel("t (token step)")
    plt.ylabel("turning angle (radians)")
    p = outdir / f"{label}_semantic_curvature.png"
    plt.tight_layout()
    plt.savefig(p, dpi=180)
    plt.close()
    return p


def persistent_homology_summary(df: pd.DataFrame, outdir: Path, label: str) -> Optional[tuple[Path, int]]:
    if ripser is None or plot_diagrams is None:
        print("[analyze_phase5] ripser/persim not available; skipping PH.")
        return None

    series = df.sort_values("t")
    texts = series["cum_text"].tolist()
    try:
        X = embed_texts_sbert(texts)
    except Exception as e:
        print(f"[analyze_phase5] Embedding failed: {e}")
        return None

    Xr = reduce_for_tda(X)
    dgms = ripser(Xr, maxdim=1, metric="euclidean")["dgms"]
    # Count H1 loops above a small persistence threshold
    num_h1_loops = 0
    try:
        H1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
        pers = H1[:, 1] - H1[:, 0] if H1.size else np.array([])
        num_h1_loops = int((pers >= 0.02).sum())
    except Exception:
        num_h1_loops = 0
    plt.figure(figsize=(6.5, 3.5))
    plot_diagrams(dgms, show=False)
    plt.title(f"Persistent homology (H0/H1) — {label}")
    ph = outdir / f"{label}_persistent_homology.png"
    plt.tight_layout()
    plt.savefig(ph, dpi=180)
    plt.close()
    return ph, num_h1_loops


# ----------------------------
# Driver
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 5 run artifacts")
    parser.add_argument("--run-dir", type=str, default=None, help="Path to a timestamped Phase 5 run directory")
    parser.add_argument("--out", type=str, default=None, help="Optional override for output directory")
    parser.add_argument("--jsonl", type=str, default=None, help="Optional explicit path to phase5_records.jsonl (overrides --run-dir)")
    args = parser.parse_args()

    output_root = Path("experiments/phase5/output")
    reports_root = Path("experiments/phase5/reports")

    # Resolve target jsonl/run_dir from args
    jsonl: Optional[Path] = None
    run_dir: Optional[Path] = None

    if args.jsonl:
        jsonl = Path(args.jsonl)
        if not jsonl.exists():
            raise FileNotFoundError(f"--jsonl path not found: {jsonl}")
        run_dir = jsonl.parent
        print(f"[analyze_phase5] Using explicit JSONL: {jsonl}")
    elif args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
    else:
        # Auto-discover: prefer timestamped dirs in output/, then reports/
        candidate = find_latest_run(output_root)
        if candidate is None:
            candidate = find_latest_run(reports_root)
        if candidate is None:
            # Deep search anywhere under experiments/phase5 for a recent phase5_records.jsonl
            roots = [output_root, reports_root, Path("experiments/phase5")] 
            found: list[Path] = []
            for root in roots:
                if root.exists():
                    for p in root.rglob("phase5_records.jsonl"):
                        found.append(p)
            if not found:
                raise FileNotFoundError(
                    "No phase5_records.jsonl found. Provide --run-dir or --jsonl, "
                    "or ensure a timestamped run exists under experiments/phase5/output/"
                )
            # pick most recently modified file
            jsonl = max(found, key=lambda p: p.stat().st_mtime)
            run_dir = jsonl.parent
            print(f"[analyze_phase5] Auto-selected by JSONL mtime: {jsonl}")
        else:
            run_dir = candidate
            print(f"[analyze_phase5] Auto-selected latest run: {run_dir}")

    if jsonl is None:
        jsonl = run_dir / "phase5_records.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"phase5_records.jsonl not found at {jsonl}")

    out = ensure_outdir(Path(args.out) if args.out else (run_dir / "analysis_phase5"))

    df = load_records(jsonl)
    groups = df.groupby(["run_id", "prompt_id"], dropna=False)
    print(f"[analyze_phase5] Loaded {len(df)} steps across {len(groups)} sequences.")
    artifacts: list[dict[str, Any]] = []

    # Optional: copy/save config snapshot into analysis folder
    cfg_map = load_cfg_yaml(run_dir)
    if cfg_map:
        try:
            (out / "config.yml").write_text(yaml.safe_dump(cfg_map, sort_keys=False))
        except Exception:
            pass

    THRESH = 0.95  # prompt-coherence half-life threshold
    for (run_id, prompt_id), g in groups:
        label = f"{run_id}_{prompt_id}"
        plot_temporal_traces(g, out, label)
        # UMAP trajectory (also capture embedding for kinematics)
        umap_res = plot_umap_trajectory(g, out, label)
        Z = None
        if isinstance(umap_res, tuple):
            _, Z = umap_res
        # Persistent homology + loop count
        ph_res = persistent_homology_summary(g, out, label)
        num_h1 = None
        if isinstance(ph_res, tuple):
            _, num_h1 = ph_res
        # Semantic kinematics from UMAP
        mean_speed = max_speed = total_path = mean_curv = max_curv = np.nan
        if Z is not None:
            kin = compute_semantic_kinematics(Z)
            speed = kin["speed"]
            curv = kin["curvature"]
            if speed.size:
                plot_semantic_velocity(speed, out, label)
                mean_speed = float(np.mean(speed))
                max_speed = float(np.max(speed))
                total_path = float(np.sum(speed))
            if curv.size:
                plot_semantic_curvature(curv, out, label)
                mean_curv = float(np.mean(curv))
                max_curv = float(np.max(curv))
        # Prompt-coherence half-life
        half_life = prompt_half_life(g.sort_values("t")["cos_to_prompt"], frac=THRESH)
        artifacts.append({
            "run_id": run_id,
            "prompt_id": prompt_id,
            "label": label,
            "model_name": str(g["model_name"].iloc[0]) if "model_name" in g.columns else "",
            "bucket": str(g["bucket"].iloc[0]) if "bucket" in g.columns else "",
            "prompt_half_life@95pct": half_life if half_life is not None else "",
            "num_h1_loops@0.02": int(num_h1) if num_h1 is not None else "",
            "mean_speed": mean_speed,
            "max_speed": max_speed,
            "total_path_len": total_path,
            "mean_curvature": mean_curv,
            "max_curvature": max_curv,
        })

    # summary CSV
    summary = out / "analysis_index.csv"
    pd.DataFrame(artifacts).to_csv(summary, index=False)
    print(f"[analyze_phase5] Saved artifacts → {out}")


if __name__ == "__main__":
    main()