#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SRB Phase III – Cross-Model Falsification Runner
Components:
  • Transfer Entropy (prompt → response)
  • NSM Primitive Coherence
  • Cross-Entropy Surprise (baseline NLL)
  • Geometry probe (UMAP trajectory placeholder, UMAP neighbor cap, Betti summary placeholder)
  • External MiniLM verifier (separate session)

This file is intentionally self-contained with TODO stubs you can replace
with your actual model/tokenizer/embedding calls.
"""

from __future__ import annotations

# ───────────────────────── Embedded default YAML ──────────────────────────────
CONFIG_YAML = """\
seed: 42
output_dir: experiments/phase3/reports
device: mps
model:
  name: gpt-4o-mini
  max_new_tokens: 256
  temperature: 0.7
  top_p: 0.95
experiment:
  runs: 200
  buckets: ["creative", "factual", "reasoning"]
  interventions: ["baseline", "embed_noise", "delay_think", "force_chain", "force_chain_strict"]
  verifier_model: "sentence-transformers/all-MiniLM-L6-v2"
metrics:
  transfer_entropy:
    history_window: 3
    estimator: ksg
  cross_entropy:
    baseline_model: "sshleifer/tiny-gpt2"
  semantic_primitives:
    nsm_list_path: experiments/phase3/vendors/nsm_primitives.txt
    weight: 1.0
  geometry:
    collect_token_embeddings: true
    umap:
      n_neighbors: 15
      min_dist: 0.1
      n_components: 2
    tda:
      compute_persistent_homology: true
      max_dim: 1
logging:
  csv: true
  jsonl: true
  plots: true
"""

# ───────────────────────────── Imports ────────────────────────────────────────
import argparse
import csv
import json
import math
import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple
import os, warnings
from tqdm import tqdm

import yaml  # pip install pyyaml

# ─────────────────────────── Data/config utils ────────────────────────────────


@dataclass
class RunConfig:
    seed: int
    output_dir: Path
    device: str
    model_name: str
    max_new_tokens: int
    temperature: float
    top_p: float
    runs: int
    buckets: list[str]
    interventions: list[str]
    verifier_model: str
    metrics: dict


class IO:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.output_dir = cfg.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, data: dict, name: str):
        from pathlib import Path
        import dataclasses
        def _json_default(o):
            if isinstance(o, Path): return str(o)
            if dataclasses.is_dataclass(o): return dataclasses.asdict(o)
            if isinstance(o, set): return list(o)
            return str(o)
        path = self.output_dir / f"{name}.json"
        path.write_text(json.dumps(data, indent=2, default=_json_default))
        return path

    def append_jsonl(self, data: dict, name: str):
        from pathlib import Path
        import dataclasses
        def _json_default(o):
            if isinstance(o, Path): return str(o)
            if dataclasses.is_dataclass(o): return dataclasses.asdict(o)
            if isinstance(o, set): return list(o)
            return str(o)
        path = self.output_dir / f"{name}.jsonl"
        with path.open("a") as f:
            f.write(json.dumps(data, default=_json_default) + "\n")
        return path

    def write_csv_rows(self, rows: list[dict], name: str):
        if not rows:
            return None
        path = self.output_dir / f"{name}.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return path


# ───────────────────────────── Prompt bank ────────────────────────────────────


@dataclass
class PromptItem:
    bucket: str
    text: str


class PromptBank:
    def __init__(self):
        # TODO: replace with your real prompt sets or file-backed bank
        self.prompts = [
            PromptItem("creative", "Write a micro-fable about a patient river."),
            PromptItem(
                "factual", "Explain the greenhouse effect in two sentences."
            ),
            PromptItem(
                "reasoning",
                "A train leaves at 2pm traveling 60 mph; a second leaves at 2:30pm traveling 80 mph in the same direction. When will it catch up? Show reasoning.",
            ),
        ]

    def sample(self, bucket: str) -> PromptItem:
        cand = [p for p in self.prompts if p.bucket == bucket]
        return random.choice(cand)


# ───────────────────────────── Interventions ──────────────────────────────────


class Intervention:
    """SRB interventions (baseline, embed_noise, delay_think, force_chain, ...)."""

    @staticmethod
    def apply(kind: str, prompt: str) -> str:
        if kind == "baseline":
            return prompt
        if kind == "embed_noise":
            # Example: add hidden instruction/noise token
            return prompt + "\n\n[NOISE: ignore this line but allocate thought budget]"
        if kind == "delay_think":
            return "(Take a deep breath and think for a moment.)\n" + prompt
        if kind == "force_chain":
            return prompt + "\n\nPlease reason step-by-step before final answer."
        if kind == "force_chain_strict":
            return prompt + "\n\nYou MUST reason in numbered steps, then provide a FINAL ANSWER block."
        # Extend with more structured interventions as needed
        return prompt


# ───────────────────────────── Metrics: TE ────────────────────────────────────


class TransferEntropy:
    """Estimate transfer entropy between sentence-level features of prompt→response.

    NOTE: This is a simple placeholder (histogram proxy). Replace with a proper
    continuous estimator (e.g., KSG) for Phase II analyses.
    """

    def __init__(self, history_window: int = 3, estimator: str = "histogram"):
        self.w = history_window
        self.estimator = estimator
        self.hist_states = deque(maxlen=self.w)

    def _quantize(self, seq: list[float], bins: int = 16) -> list[int]:
        if not seq:
            return []
        lo, hi = min(seq), max(seq)
        if hi == lo:
            return [0] * len(seq)
        return [
            min(bins - 1, int((x - lo) / (hi - lo + 1e-12) * bins)) for x in seq
        ]

    def compute(self, prompt_feats: list[float], response_feats: list[float]) -> float:
        qp = self._quantize(prompt_feats)
        qr = self._quantize(response_feats)
        joint: dict[tuple[int, int], int] = {}
        for a, b in zip(qp, qr):
            joint[(a, b)] = joint.get((a, b), 0) + 1
        pa: dict[int, int] = {}
        pb: dict[int, int] = {}
        for a, b in joint:
            pa[a] = pa.get(a, 0) + joint[(a, b)]
            pb[b] = pb.get(b, 0) + joint[(a, b)]
        n = max(1, sum(joint.values()))
        te = 0.0
        for (a, b), c in joint.items():
            p_ab = c / n
            p_a = pa[a] / n
            p_b = pb[b] / n
            if p_ab > 0 and p_a > 0 and p_b > 0:
                te += p_ab * math.log((p_ab / (p_a * p_b + 1e-12)) + 1e-12)
        return max(0.0, te)


# ─────────────────────── Metrics: Cross-Entropy NLL ───────────────────────────


class CrossEntropySurprise:
    """Cross-entropy (negative log-likelihood) of a response under a baseline model.

    Tries to load a tiny causal LM via Hugging Face (CPU). If unavailable, falls back to a
    deterministic heuristic so the pipeline always runs.
    """

    def __init__(self, baseline_model_name: str, device: str = "cpu"):
        self.name = baseline_model_name
        self._hf_ok = False
        self._tok = None
        self._lm = None
        self._device = device
        # Detect MPS availability if requested
        if device == "mps":
            try:
                import torch
                if hasattr(torch, "has_mps") and torch.has_mps:
                    self._device = "mps"
                else:
                    self._device = "cpu"
            except Exception:
                self._device = "cpu"
        try:
            # Lazy import to avoid hard dependency if user hasn't installed transformers/torch
            from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
            import torch  # type: ignore

            self._tok = AutoTokenizer.from_pretrained(self.name)
            self._lm = AutoModelForCausalLM.from_pretrained(self.name)
            self._lm.to(self._device)
            self._lm.eval()
            self._hf_ok = True
            print(f"[SRB][HF] CrossEntropySurprise using HF model: {self.name} (device={self._device})")
        except Exception as e:
            # Fallback mode (still deterministic)
            print(f"[SRB][HF] CrossEntropySurprise fallback (stub NLL): {e}")

    def nll(self, response_text: str) -> float:
        """Return average per-token NLL (lower is 'less surprising' under the baseline)."""
        if self._hf_ok and self._tok is not None and self._lm is not None:
            try:
                import torch  # type: ignore

                with torch.no_grad():
                    enc = self._tok(
                        response_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=1024,
                    )
                    enc = {k: v.to(self._device) for k, v in enc.items()}
                    # Standard LM loss: labels = input_ids (next-token prediction)
                    out = self._lm(**enc, labels=enc["input_ids"])
                    # HF returns mean loss over tokens
                    loss = float(out.loss.detach().cpu().item())
                    return loss
            except Exception as e:
                print(f"[SRB][HF] NLL computation failed, using stub: {e}")

        # Stub fallback: simple, length-scaled value (keeps pipeline running)
        # Tuned to be in a plausible numeric range (e.g., ~0.5–5.0)
        L = max(1, len(response_text))
        unique = len(set(response_text.split()))
        return 1.5 + 0.002 * L - 0.001 * unique


# ─────────────── Metrics: NSM Primitive Coherence (starter) ───────────────────


class SemanticPrimitiveCoherence:
    """NSM primitive alignment checker.
    Counts presence/coverage of primitive lexeme set as a cheap proxy.

    TODO ideas:
      • track roles/relations (dependency parse)
      • weight primitives by expected distribution per bucket
      • sentence-level pass/fail against simple templates
    """

    def __init__(self, nsm_list_path: str):
        self.primitives = self._load(nsm_list_path)

    def _load(self, path: str) -> set[str]:
        try:
            return set(Path(path).read_text().split())
        except Exception:
            # Minimal seed list; replace with canonical NSM inventory in vendors/
            return {
                "I",
                "YOU",
                "SOMEONE",
                "PEOPLE",
                "SAY",
                "THINK",
                "KNOW",
                "DO",
                "HAPPEN",
                "BECAUSE",
                "IF",
                "GOOD",
                "BAD",
                "BIG",
                "SMALL",
                "MANY",
                "FEW",
                "BEFORE",
                "AFTER",
                "HERE",
                "NOW",
                "THIS",
                "SAME",
                "OTHER",
            }

    def score(self, text: str) -> float:
        words = set(w.strip(".,!?;:\"").upper() for w in text.split())
        hits = words.intersection(self.primitives)
        # Simple coverage ratio as a starting point
        return len(hits) / max(1, len(self.primitives))


# ───────────────────────── Geometry/TDA probe (stubs) ─────────────────────────


class GeometryProbe:
    """Collect token embeddings → reduce with UMAP → compute simple TDA proxies.

    Tries real UMAP + ripser if installed; falls back to stubs gracefully.
    """
    def __init__(self, umap_cfg: dict, tda_cfg: dict):
        self.umap_cfg = umap_cfg
        self.tda_cfg = tda_cfg
        # Safe optional imports
        try:
            import umap  # noqa: F401
            self._umap_available = True
        except Exception:
            self._umap_available = False

        try:
            import ripser  # noqa: F401
            self._ripser_available = True
        except Exception:
            self._ripser_available = False

    def collect_embeddings(self, tokens: list[str]) -> list[list[float]]:
        # TODO: replace with real model hidden states
        return [[hash(t) % 101 / 100.0] for t in tokens]

    def umap_project(self, X: list[list[float]]) -> list[tuple[float, float]]:
        # Cap n_neighbors at min(cap_in, len(X)-2)
        n_neighbors_in = self.umap_cfg.get("n_neighbors", 15)
        if len(X) < 3:
            # Not enough samples for UMAP, return stub
            return [(float(i), row[0]) for i, row in enumerate(X)]
        cap = min(n_neighbors_in, len(X) - 2)
        if self._umap_available:
            try:
                import umap
                reducer = umap.UMAP(
                    n_neighbors=cap,
                    min_dist=self.umap_cfg.get("min_dist", 0.1),
                    n_components=self.umap_cfg.get("n_components", 2),
                    random_state=42,
                )
                X2d = reducer.fit_transform(X)
                return [(float(x), float(y)) for x, y in X2d]
            except Exception:
                pass  # fall through to stub
        # Stub fallback
        return [(float(i), row[0]) for i, row in enumerate(X)]

    def betti_summary(self, X2d: list[tuple[float, float]]) -> dict:
        if self._ripser_available:
            try:
                from ripser import ripser
                import numpy as np
                pts = np.array(X2d)
                res = ripser(pts, maxdim=self.tda_cfg.get("max_dim", 1))
                dgms = res.get("dgms", [])
                betti0 = len(dgms[0]) if len(dgms) > 0 else 0
                betti1 = len(dgms[1]) if len(dgms) > 1 else 0
                return {"betti0": betti0, "betti1": betti1}
            except Exception:
                pass  # fall through
        # Stub fallback
        return {"betti0": 1, "betti1": 0}


# ───────────────────────── External verifier (stub) ───────────────────────────


class SimpleBERTVerifier:
    """External low-vocab BERT-style sanity checker.

    Attempts to load a small encoder (e.g., 'prajjwal1/bert-tiny') and compute cosine
    similarity between pooled CLS embeddings. Falls back to a lexical Jaccard overlap
    if HF isn't available. Keep this verifier *logically* outside the main design loop.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        self.name = model_name
        self._hf_ok = False
        self._tok = None
        self._enc = None
        self._device = device
        # Detect MPS availability if requested
        if device == "mps":
            try:
                import torch
                if hasattr(torch, "has_mps") and torch.has_mps:
                    self._device = "mps"
                else:
                    self._device = "cpu"
            except Exception:
                self._device = "cpu"
        try:
            from transformers import AutoTokenizer, AutoModel  # type: ignore
            import torch  # type: ignore

            self._tok = AutoTokenizer.from_pretrained(self.name)
            self._enc = AutoModel.from_pretrained(self.name)
            self._enc.to(self._device)
            self._enc.eval()
            self._hf_ok = True
            print(f"[SRB][HF] Verifier using HF encoder: {self.name} (device={self._device})")
        except Exception as e:
            print(f"[SRB][HF] Verifier fallback (stub similarity): {e}")

    def _embed(self, text: str):
        """Return pooled CLS embedding (1, D) if HF is available; else None."""
        if not self._hf_ok or self._tok is None or self._enc is None:
            return None
        try:
            import torch  # type: ignore

            with torch.no_grad():
                enc = self._tok(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                enc = {k: v.to(self._device) for k, v in enc.items()}
                out = self._enc(**enc)
                # CLS pooled rep: last_hidden_state[:, 0, :]
                cls = out.last_hidden_state[:, 0, :]  # (1, D)
                # Normalize for stable cosine later
                cls = torch.nn.functional.normalize(cls, dim=-1)
                return cls.cpu()
        except Exception:
            return None

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity in [0,1] when HF available; otherwise Jaccard fallback."""
        if self._hf_ok:
            try:
                import torch  # type: ignore

                ea = self._embed(a)
                eb = self._embed(b)
                if ea is not None and eb is not None:
                    cos = torch.nn.functional.cosine_similarity(ea, eb).item()
                    # Map from [-1,1] → [0,1]
                    return 0.5 * (cos + 1.0)
            except Exception:
                pass

        # Fallback: lexical Jaccard similarity over lowercased word sets
        set_a = set(w.strip(".,!?;:\"'()[]{}").lower() for w in a.split() if w.strip())
        set_b = set(w.strip(".,!?;:\"'()[]{}").lower() for w in b.split() if w.strip())
        if not set_a and not set_b:
            return 0.0
        inter = len(set_a & set_b)
        union = max(1, len(set_a | set_b))
        return inter / union


# ───────────────────────────── Runner ─────────────────────────────────────────


class Phase3Runner:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.io = IO(cfg)
        # Timestamped subdirectory for output
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.io.output_dir = self.io.output_dir / f"run_{stamp}"
        self.io.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompts = PromptBank()

        te_cfg = cfg.metrics.get("transfer_entropy", {})
        xent_cfg = cfg.metrics.get("cross_entropy", {})
        nsm_cfg = cfg.metrics.get("semantic_primitives", {})
        geo_cfg = cfg.metrics.get("geometry", {})

        self.te = TransferEntropy(**te_cfg)
        self.xent = CrossEntropySurprise(xent_cfg.get("baseline_model", cfg.model_name), device=cfg.device)
        self.nsm = SemanticPrimitiveCoherence(
            nsm_cfg.get("nsm_list_path", "experiments/phase3/vendors/nsm_primitives.txt")
        )
        self.geo = GeometryProbe(geo_cfg.get("umap", {}), geo_cfg.get("tda", {}))
        self.verifier = SimpleBERTVerifier(cfg.verifier_model, device=cfg.device)

        # Capabilities flags
        self.capabilities = dict(
            umap_used=getattr(self.geo, "_umap_available", False),
            ripser_used=getattr(self.geo, "_ripser_available", False),
            hf_nll_used=getattr(self.xent, "_hf_ok", False),
            hf_verifier_used=getattr(self.verifier, "_hf_ok", False),
            model_name=cfg.model_name,
            verifier_model=cfg.verifier_model,
            device=cfg.device,
            temperature=cfg.temperature,
        )

    # TODO: Replace this with your actual model invocation
    def _gen(self, prompt: str) -> str:
        return f"[MODEL OUTPUT] {prompt[:48]} …"

    # TODO: Replace with real feature extraction (length, type-token ratio, entropy, etc.)
    def _features_from_text(self, text: str) -> list[float]:
        alpha = sum(c.isalpha() for c in text)
        return [len(text), alpha / max(1, len(text))]

    def run(self):
        random.seed(self.cfg.seed)
        rows: list[dict] = []
        timestamp = datetime.utcnow().isoformat()

        self.io.write_json({"config": self.cfg.__dict__, "timestamp": timestamp, "capabilities": self.capabilities}, name="phase3_config")

        for bucket in self.cfg.buckets:
            for i in tqdm(range(self.cfg.runs), desc=f"bucket={bucket}", leave=False):
                for inter in self.cfg.interventions:
                    base_prompt = self.prompts.sample(bucket)
                    prompt_i = Intervention.apply(inter, base_prompt.text)
                    response = self._gen(prompt_i)

                    # Metrics
                    pf = self._features_from_text(prompt_i)
                    rf = self._features_from_text(response)
                    te_val = self.te.compute(pf, rf)
                    nll_val = self.xent.nll(response)
                    nsm_val = self.nsm.score(response)

                    # Geometry/TDA (token-level)
                    tokens = response.split()
                    emb = self.geo.collect_embeddings(tokens)
                    traj2d = self.geo.umap_project(emb)
                    betti = self.geo.betti_summary(traj2d)

                    # External verification (separate session/model ideally)
                    ver_sim = self.verifier.similarity(prompt_i, response)

                    rec = {
                        "bucket": bucket,
                        "run": i,
                        "intervention": inter,
                        "prompt": base_prompt.text,
                        "prompt_i": prompt_i,
                        "response": response,
                        "transfer_entropy": te_val,
                        "cross_entropy_nll": nll_val,
                        "nsm_coherence": nsm_val,
                        "verifier_similarity": ver_sim,
                        "betti0": betti.get("betti0"),
                        "betti1": betti.get("betti1"),
                    }

                    rows.append(rec)
                    self.io.append_jsonl(rec, name="phase3_records")

        self.io.write_csv_rows(rows, name="phase3_summary")
        print(f"Saved {len(rows)} records → {self.cfg.output_dir}")


# ─────────────────────────── CLI entry point ──────────────────────────────────


def load_cfg_from_yaml(yaml_text: str) -> RunConfig:
    cfg = yaml.safe_load(yaml_text)
    return RunConfig(
        seed=cfg["seed"],
        output_dir=Path(cfg["output_dir"]),
        device=cfg.get("device", "cpu"),
        model_name=cfg["model"]["name"],
        max_new_tokens=cfg["model"]["max_new_tokens"],
        temperature=cfg["model"]["temperature"],
        top_p=cfg["model"]["top_p"],
        runs=cfg["experiment"]["runs"],
        buckets=cfg["experiment"]["buckets"],
        interventions=cfg["experiment"]["interventions"],
        verifier_model=cfg["experiment"]["verifier_model"],
        metrics=cfg["metrics"],
    )


if __name__ == "__main__":
    # Set environment variables and warnings before parsing args
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    warnings.filterwarnings("ignore", message="n_jobs value 1 overridden*")

    parser = argparse.ArgumentParser(description="SRB Phase III Runner")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (optional; uses embedded defaults if omitted)",
    )
    parser.add_argument("--runs", type=int, default=None, help="Override number of runs")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--verifier", type=str, default=None, help="Override verifier model")
    parser.add_argument("--temperature", type=float, default=None, help="Override model temperature")
    parser.add_argument("--device", type=str, choices=["cpu", "mps"], default=None, help="Device to use (cpu|mps)")
    parser.add_argument("--output_suffix", type=str, default=None, help="Append suffix to output dir")
    args = parser.parse_args()

    # Config loading logic
    cfg_text = None
    config_paths = []
    if args.config:
        config_paths.append(args.config)
    else:
        config_paths.append("experiments/phase3/config/defaults.yaml")
    found = False
    for p in config_paths:
        try:
            cfg_text = Path(p).read_text()
            found = True
            break
        except Exception:
            pass
    if not found:
        cfg_text = CONFIG_YAML

    cfg = load_cfg_from_yaml(cfg_text)
    # CLI overrides
    if args.runs is not None:
        cfg.runs = args.runs
    if args.model is not None:
        cfg.model_name = args.model
    if args.verifier is not None:
        cfg.verifier_model = args.verifier
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.device is not None:
        cfg.device = args.device

    runner = Phase3Runner(cfg)
    # If output_suffix, append to timestamped directory and ensure it exists
    if args.output_suffix:
        runner.io.output_dir = Path(str(runner.io.output_dir) + f"_{args.output_suffix}")
        runner.io.output_dir.mkdir(parents=True, exist_ok=True)
    runner.run()