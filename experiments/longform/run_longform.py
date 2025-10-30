#!/usr/bin/env python3
"""
Long-form SRB experiment runner.

Reads a "bucket" file of prompts (separated by blank lines), runs SRB dynamic
decoding for each prompt, and writes per-step traces (trace.csv) and a
summary.json into per-run folders under reports/longform/YYYYMMDD_HHMM/.

Assumes the following functions/classes exist:
- srb.srb_wrapper.load_model(model_id: str, device: str) -> (model, tokenizer, device_str)
- srb.srb_wrapper.run_srb_dynamic(model, tok, prompt: str, cfg: dict) -> dict
  The return dict is expected to contain:
    - "text": final generated text
    - "per_step_data": List[dict] with keys including:
        step, token, text, entropy, coherence, resonance, d_resonance, temperature, top_p
    - "entropies", "coherences", "resonances": lists of floats
    - summary aggregates like entropy_mean, entropy_final, etc. (if available)
- srb.tracing.TraceLogger for writing trace.csv and summary.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import yaml
import random
import numpy as np
try:
    import torch
except Exception:
    torch = None

from srb.tracing import TraceLogger, save_summary
from srb.metrics import aggregate_run_metrics
from srb.srb_wrapper import load_model, run_srb_dynamic
from srb.plotting import plot_resonance, plot_entropy_coherence

# -----------------------------
# Bucket detection
# -----------------------------

def detect_bucket_name(bucket_path: Path) -> str:
    """Infer bucket name from the bucket file path/name.
    Looks for known tokens in the path; otherwise derives from a 'bucket_*.txt' stem.
    Returns one of {"creative", "multilingual", "science", "unknown"}.
    """
    tokens = ("creative", "multilingual", "science")
    path_str = str(bucket_path).lower()
    for t in tokens:
        if t in path_str:
            return t
    # Fallback: derive from file name like 'bucket_science.txt'
    stem = bucket_path.stem.lower()
    if stem.startswith("bucket_"):
        return stem.split("bucket_", 1)[1]
    return "unknown"

# -----------------------------
# Utilities
# -----------------------------

def read_bucket(path: Path) -> List[str]:
    """Read prompts separated by one or more blank lines."""
    txt = path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in txt.split("\n\n") if b.strip()]
    return blocks

def ensure_out_root(out_root: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M")
    run_root = out_root / ts
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root

def set_seed(seed: int) -> None:
    """Set python, numpy, and torch RNG seeds if available."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

def write_final_index(run_root: Path, summaries: List[Dict[str, Any]]) -> None:
    """Write an aggregated summary.json at the root."""
    if not summaries:
        return
    agg = {
        "runs": len(summaries),
        "avg_entropy_mean": sum(s.get("entropy_mean", 0.0) for s in summaries) / len(summaries),
        "avg_coherence_mean": sum(s.get("coherence_mean", 0.0) for s in summaries) / len(summaries),
        "avg_R_mean": sum(s.get("R_mean", 0.0) for s in summaries) / len(summaries),
        "avg_R_collapse_steps": sum(s.get("R_collapse_steps", 0) for s in summaries) / len(summaries),
    }
    (run_root / "summary.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Run SRB long-form prompts bucket.")
    ap.add_argument("--bucket", required=True, help="Path to a text file with prompts separated by blank lines.")
    ap.add_argument("--bucket_name", default=None, help="Optional explicit bucket name (creative|multilingual|science). If omitted, inferred from --bucket path.")
    ap.add_argument("--out_root", default="reports/longform", help="Output root directory.")
    ap.add_argument("--device", default="auto", help="Device: auto | cpu | cuda | mps")
    ap.add_argument("--model_id", default="meta-llama/Llama-3-8B-Instruct", help="HF model id.")
    ap.add_argument("--config", default="experiments/longform/config.longform.yaml", help="YAML config file for SRB parameters.")
    # Generation / SRB config (can be overridden on CLI)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature_base", type=float, default=0.8)
    ap.add_argument("--top_p_base", type=float, default=0.95)
    ap.add_argument("--entropy_top_k", type=int, default=5)
    ap.add_argument("--resonance_alpha", type=float, default=1.0)
    ap.add_argument("--completion_eps", type=float, default=0.01)
    ap.add_argument("--completion_window", type=int, default=5)
    ap.add_argument("--min_tokens", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--reflect_on_drop", action="store_true")
    ap.add_argument("--reflect_drop_threshold", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=None, help="Base random seed; if None, uses time-based seed. Each run uses seed = base + index.")
    ap.add_argument("--intervention", type=str, choices=["none", "logit_shuffle", "embed_noise"], default="none", help="Intervention type to apply during generation.")
    args = ap.parse_args()

    base_seed = args.seed if args.seed is not None else int(time.time()) % 1_000_000

    bucket_path = Path(args.bucket)
    out_root_base = Path(args.out_root)
    # Determine bucket name (CLI override wins, else infer from path)
    bucket_name = (args.bucket_name or detect_bucket_name(bucket_path)).strip().lower()
    if bucket_name not in {"creative", "multilingual", "science", "unknown"}:
        bucket_name = "unknown"
    out_root = out_root_base / bucket_name

    run_root = ensure_out_root(out_root)

    # Load YAML config
    cfg_yaml = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        try:
            cfg_yaml = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            print(f"[SRB] Warning: failed to read YAML config {cfg_path}: {e}")
            cfg_yaml = {}
    else:
        print(f"[SRB] Note: config file not found at {cfg_path}, using CLI/defaults.")

    prompts = read_bucket(bucket_path)
    if not prompts:
        raise SystemExit(f"No prompts found in bucket: {bucket_path}")

    # Resolve model_id and device with precedence: CLI > YAML > defaults
    parser_defaults = {
        "model_id": "meta-llama/Llama-3-8B-Instruct",
        "device": "auto",
    }
    model_id = args.model_id if args.model_id != parser_defaults["model_id"] else cfg_yaml.get("model_id", args.model_id)
    device_arg = args.device if args.device != parser_defaults["device"] else cfg_yaml.get("device", args.device)

    print(f"[SRB] Loading model: {model_id} on device={device_arg}")
    model, tok, device = load_model(model_id, device=device_arg)
    print(f"[SRB] Loaded. Bucket={bucket_name}. Running {len(prompts)} prompts → {run_root}")

    # Start with YAML config, then override with CLI-provided values where they differ from defaults
    cfg = dict(cfg_yaml) if isinstance(cfg_yaml, dict) else {}

    # Ensure mandatory fields have safe defaults if YAML missing
    cfg.setdefault("max_new_tokens", args.max_new_tokens)
    cfg.setdefault("temperature_base", args.temperature_base)
    cfg.setdefault("top_p_base", args.top_p_base)
    cfg.setdefault("entropy_top_k", args.entropy_top_k)
    cfg.setdefault("resonance_alpha", args.resonance_alpha)
    cfg.setdefault("completion_eps", args.completion_eps)
    cfg.setdefault("completion_window", args.completion_window)
    cfg.setdefault("min_tokens", args.min_tokens)
    cfg.setdefault("max_tokens", args.max_tokens)
    cfg.setdefault("reflect_on_drop", bool(args.reflect_on_drop))
    cfg.setdefault("reflect_drop_threshold", args.reflect_drop_threshold)

    # Apply CLI overrides when they differ from argparse defaults (user explicitly set)
    parser_default_map = {
        "max_new_tokens": 512,
        "temperature_base": 0.8,
        "top_p_base": 0.95,
        "entropy_top_k": 5,
        "resonance_alpha": 1.0,
        "completion_eps": 0.01,
        "completion_window": 5,
        "min_tokens": 64,
        "max_tokens": 2048,
        "reflect_on_drop": False,
        "reflect_drop_threshold": 0.10,
    }
    for key in parser_default_map:
        arg_val = getattr(args, key, None)
        if arg_val is not None and arg_val != parser_default_map[key]:
            cfg[key] = arg_val

    # Add intervention to config
    cfg["intervention"] = args.intervention

    # Attach resolved device
    cfg["device"] = device_arg

    summaries: List[Dict[str, Any]] = []

    for i, prompt in enumerate(prompts):
        run_dir = run_root / f"run_{i:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Derive per-run seed and set RNGs
        run_seed = int(base_seed + i)
        set_seed(run_seed)
        cfg["seed"] = run_seed

        print(f"[SRB] ▶ Running prompt {i+1}/{len(prompts)} (seed={run_seed}, intervention={args.intervention}) → {run_dir}")

        # Run SRB dynamic decoding
        result: Dict[str, Any] = run_srb_dynamic(model, tok, prompt, cfg)

        # Trace logging
        logger = TraceLogger(run_dir)
        steps = result.get("per_step_data", [])
        for row in steps:
            logger.log_step(**{
                "step": row.get("step"),
                "token": row.get("token"),
                "text_so_far": row.get("text") or row.get("text_so_far", ""),
                "entropy": float(row.get("entropy", 0.0)),
                "coherence": float(row.get("coherence", 1.0)),
                "resonance": float(row.get("resonance", 0.0)),
                "d_resonance": float(row.get("d_resonance", 0.0)),
                "temperature": float(row.get("temperature", cfg["temperature_base"])),
                "top_p": float(row.get("top_p", cfg["top_p_base"])),
            })

        # Aggregate and persist summary
        if steps and (not result.get("R_mean")):
            ent = [float(r.get("entropy", 0.0)) for r in steps]
            coh = [float(r.get("coherence", 1.0)) for r in steps]
            res = [float(r.get("resonance", 0.0)) for r in steps]
            summary = aggregate_run_metrics(ent, coh, res)
            summary["seed"] = run_seed
        else:
            summary = {
                "text_tokens": len(steps),
                "entropy_mean": result.get("entropy_mean", 0.0),
                "entropy_std": result.get("entropy_std", 0.0),
                "entropy_final": result.get("entropy_final", result.get("final_entropy", 0.0)),
                "coherence_mean": result.get("avg_coherence", result.get("coherence_mean", 0.0)),
                "coherence_final": result.get("final_coherence", result.get("coherence_final", 0.0)),
                "R_mean": result.get("avg_resonance", result.get("R_mean", 0.0)),
                "R_final": result.get("final_resonance", result.get("R_final", 0.0)),
                "R_collapse_steps": result.get("R_collapse_steps", 0),
                "seed": run_seed,
            }

        # Include intervention in summary
        summary["intervention"] = args.intervention

        # Include final text for reference
        final_text = result.get("text", "")
        (run_dir / "output.txt").write_text(final_text, encoding="utf-8")
        (run_dir / "seed.txt").write_text(str(run_seed), encoding="utf-8")

        # Plots
        try:
            trace_csv = run_dir / "trace.csv"
            plot_resonance(trace_csv, run_dir / "resonance_plot.png", title=f"Run {i:02d} Resonance")
            plot_entropy_coherence(trace_csv, run_dir / "entropy_plot.png", title=f"Run {i:02d}")
        except Exception as e:
            print(f"[SRB] Plotting failed for run {i:02d}: {e}")

        logger.finalize(summary)
        summaries.append(summary)

    write_final_index(run_root, summaries)
    print(f"[SRB] ✅ Done. Root summary: {run_root/'summary.json'}")

if __name__ == "__main__":
    main()
