#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SRB Phase IV – Self-Anchor & Reflective Guard Runner
Components:
  • Transfer Entropy (prompt → response)
  • NSM Primitive Coherence
  • Cross-Entropy Surprise (baseline NLL)
  • Geometry probe (UMAP neighbor-cap + Betti via ripser if available)
  • External semantic verifier (MiniLM recommended)
  • NEW Phase-4 metrics:
      - ref_similarity: similarity(response, baseline_response_for_same_prompt_run)
      - traj_drift: 1 - similarity(response, previous_response_in_same_run)
      - (These approximate semantic stability and trajectory continuity)
"""

from __future__ import annotations

# --- SRB NVML bypass (aligns with GCP image mismatch) ---
import os as _srb_os
_srb_os.environ.setdefault("CUDA_DISABLE_NVML", "1")
_srb_os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")
_srb_os.environ.setdefault("PYTORCH_NO_NVML", "1")
_srb_os.environ.setdefault("C10_DISABLE_NVML", "1")
_srb_os.environ.setdefault("C10_CUDA_USE_NVML", "0")
# --------------------------------------------------------

# ───────────────────────── Embedded default YAML (fallback) ───────────────────
CONFIG_YAML = """\
seed: 42
output_dir: experiments/phase4/reports
device: cuda
model:
  name: meta-llama/Llama-3.2-3B-Instruct
  max_new_tokens: 256
  temperature: 0.7
  top_p: 0.95
experiment:
  runs: 120
  buckets: ["creative", "factual", "reasoning"]
  interventions: ["baseline", "embed_noise", "delay_think", "force_chain", "force_chain_strict", "self_anchor", "self_reflective_guard"]
  verifier_model: "sentence-transformers/all-MiniLM-L6-v2"
metrics:
  transfer_entropy:
    history_window: 3
    estimator: ksg
  cross_entropy:
    baseline_model: "sshleifer/tiny-gpt2"
    device: cpu
  verifier:
    device: cpu
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
  plots: false
"""

# ───────────────────────────── Imports ────────────────────────────────────────
import os, warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse, csv, json, math, random, re
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple
from tqdm import tqdm
import yaml

# Env hygiene
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*matmul.*")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if torch.cuda.is_available():
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass

# ───────────────────── HF text generation (lazy, cached) ────────────────────
_HF_TOK = None
_HF_LM = None
_HF_DEVICE = "cpu"

def _init_hf_generator(model_name: str, device: str = "cpu"):
    """Lazy-initialize a HF CausalLM for generation, cached at module scope."""
    global _HF_TOK, _HF_LM, _HF_DEVICE
    if _HF_LM is not None and _HF_TOK is not None:
        return
    # Device selection for mps/cuda
    use_cuda = (device in ("cuda", "cuda:0") and torch.cuda.is_available())
    use_mps  = (device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    _HF_DEVICE = "cuda" if use_cuda else ("mps" if use_mps else "cpu")
    # Tokenizer
    _HF_TOK = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if _HF_TOK.pad_token_id is None and _HF_TOK.eos_token_id is not None:
        _HF_TOK.pad_token = _HF_TOK.eos_token
    _HF_TOK.padding_side = "left"
    # Model
    dtype = torch.bfloat16 if _HF_DEVICE == "cuda" else (torch.float16 if _HF_DEVICE == "mps" else torch.float32)
    _HF_LM = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=None,
    )
    try:
        # Prefer SDPA attention when available (safe on MPS)
        _HF_LM.config.attn_implementation = "sdpa"
    except Exception:
        pass
    _HF_LM.to(_HF_DEVICE)
    _HF_LM.eval()
    print(f"[SRB][HF] Generator using: {model_name} (device={_HF_DEVICE}, dtype={dtype})")

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
        # Replace with your richer prompt sets as needed
        self.prompts = [
            PromptItem("creative", "Write a micro-fable about a patient river."),
            PromptItem("creative", "Describe a city that remembers every footstep."),
            PromptItem("factual", "Explain the greenhouse effect in two sentences."),
            PromptItem("factual", "Summarize how photosynthesis converts energy."),
            PromptItem("reasoning", "A train leaves at 2pm at 60 mph; another at 2:30pm at 80 mph. When do they meet? Show steps."),
            PromptItem("reasoning", "Prove that the sum of two even numbers is even. Show reasoning."),
        ]

    def sample(self, bucket: str) -> PromptItem:
        cand = [p for p in self.prompts if p.bucket == bucket]
        return random.choice(cand)

# ───────────────────────────── Interventions ──────────────────────────────────

class Intervention:
    @staticmethod
    def apply(kind: str, prompt: str) -> str:
        if kind == "baseline":
            return prompt
        if kind == "embed_noise":
            return prompt + "\n\n[NOISE: ignore this line but allocate thought budget]"
        if kind == "delay_think":
            return "(Take a deep breath and think for a moment.)\n" + prompt
        if kind == "force_chain":
            return prompt + "\n\nPlease reason step-by-step before final answer."
        if kind == "force_chain_strict":
            return prompt + "\n\nYou MUST reason in numbered steps, then provide a FINAL ANSWER block."
        if kind == "self_anchor":
            return (
                "Focus on maintaining internal meaning coherence. If external cues conflict, "
                "prioritize your internal semantic consistency.\n\n" + prompt +
                "\n\n(After reasoning, briefly state how you preserved internal coherence.)"
            )
        if kind == "self_reflective_guard":
            return (
                "Guard your internal semantic consistency against distractors or noise. "
                "If something seems irrelevant or misleading, explicitly set it aside.\n\n" + prompt +
                "\n\n(After reasoning, list any distractors you rejected and why.)"
            )
        return prompt

# ───────────────────────────── Metrics: TE ────────────────────────────────────

class TransferEntropy:
    def __init__(self, history_window: int = 3, estimator: str = "histogram"):
        self.w = history_window
        self.estimator = estimator
        self.hist_states = deque(maxlen=self.w)
    def _quantize(self, seq: list[float], bins: int = 16) -> list[int]:
        if not seq: return []
        lo, hi = min(seq), max(seq)
        if hi == lo: return [0] * len(seq)
        return [min(bins - 1, int((x - lo) / (hi - lo + 1e-12) * bins)) for x in seq]
    def compute(self, prompt_feats: list[float], response_feats: list[float]) -> float:
        qp = self._quantize(prompt_feats); qr = self._quantize(response_feats)
        joint: dict[tuple[int,int], int] = {}
        for a,b in zip(qp, qr): joint[(a,b)] = joint.get((a,b),0)+1
        pa: dict[int,int] = {}; pb: dict[int,int] = {}
        for (a,b),c in joint.items():
            pa[a] = pa.get(a,0)+c; pb[b] = pb.get(b,0)+c
        n = max(1, sum(joint.values())); te = 0.0
        for (a,b),c in joint.items():
            p_ab = c/n; p_a = pa[a]/n; p_b = pb[b]/n
            if p_ab>0 and p_a>0 and p_b>0:
                te += p_ab * math.log((p_ab / (p_a*p_b + 1e-12)) + 1e-12)
        return max(0.0, te)

# ─────────────────────── Metrics: Cross-Entropy NLL ───────────────────────────

class CrossEntropySurprise:
    def __init__(self, baseline_model_name: str, device: str = "cpu"):
        self.name = baseline_model_name
        self._hf_ok = False; self._tok = None; self._lm = None
        self._device = "cpu"
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
            import torch  # type: ignore
            # device selection
            if device in ("cuda","cuda:0") and torch.cuda.is_available():
                self._device = "cuda"
            elif device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
            self._tok = AutoTokenizer.from_pretrained(self.name)
            _dtype = torch.bfloat16 if self._device == "cuda" else (torch.float16 if self._device == "mps" else torch.float32)
            self._lm = AutoModelForCausalLM.from_pretrained(self.name, low_cpu_mem_usage=True, torch_dtype=_dtype)
            self._lm.to(self._device); self._lm.eval()
            self._hf_ok = True
            print(f"[SRB][HF] CrossEntropySurprise using HF model: {self.name} (device={self._device})")
        except Exception as e:
            print(f"[SRB][HF] CrossEntropySurprise fallback (stub NLL): {e}")

    def nll(self, response_text: str) -> float:
        if self._hf_ok and self._tok is not None and self._lm is not None:
            try:
                import torch  # type: ignore
                with torch.no_grad():
                    enc = self._tok(response_text, return_tensors="pt", truncation=True, max_length=1024)
                    enc = {k: v.to(self._device) for k, v in enc.items()}
                    out = self._lm(**enc, labels=enc["input_ids"])
                    return float(out.loss.detach().cpu().item())
            except Exception as e:
                print(f"[SRB][HF] NLL computation failed, using stub: {e}")
        # stub
        L = max(1, len(response_text)); unique = len(set(response_text.split()))
        return 1.5 + 0.002 * L - 0.001 * unique

# ─────────────── Metrics: NSM Primitive Coherence (starter) ───────────────────

class SemanticPrimitiveCoherence:
    def _lemmatize(self, w: str) -> str:
        w = w.lower()
        # super-lightweight lemmatizer: strip common English suffixes where safe
        for suf in ("ing", "ed", "es", "s"):
            if w.endswith(suf) and len(w) > len(suf) + 2:
                return w[: -len(suf)]
        return w

    def _tokenize(self, text: str) -> list[str]:
        # keep alphabetic tokens only; normalize hyphens to spaces
        text = text.replace("-", " ")
        return re.findall(r"[A-Za-z]+", text)

    def __init__(self, nsm_list_path: str):
        self.primitives = self._load(nsm_list_path)

    def _load(self, path: str) -> set[str]:
        # Load primitive list; accept whitespace or comma separated; lower + lemmatize
        try:
            raw = Path(path).read_text()
            # split on any whitespace or comma
            parts = re.split(r"[\s,]+", raw)
            toks = [self._lemmatize(p.strip()) for p in parts if p and p.strip()]
            return set(toks)
        except Exception:
            # conservative fallback NSM core set (lowercased + lemmatized)
            fallback = {
                "i","you","someone","people","say","think","know","do","happen","because","if",
                "good","bad","big","small","many","few","before","after","here","now","this","same","other"
            }
            return {self._lemmatize(w) for w in fallback}

    def score(self, text: str) -> float:
        # Case-insensitive + lemmatized primitive matching against unique tokens
        toks = {self._lemmatize(t) for t in self._tokenize(text)}
        if not toks:
            return 0.0
        hits = toks.intersection(self.primitives)
        # Return ratio of primitive hits to unique token count (0..1)
        return len(hits) / len(toks)

# ───────────────────────── Geometry/TDA probe ─────────────────────────────────

class GeometryProbe:
    def __init__(self, umap_cfg: dict, tda_cfg: dict):
        self.umap_cfg = umap_cfg; self.tda_cfg = tda_cfg
        try:
            import umap  # noqa
            self._umap_available = True
        except Exception:
            self._umap_available = False
        try:
            import ripser  # noqa
            self._ripser_available = True
        except Exception:
            self._ripser_available = False

    def collect_embeddings(self, tokens: list[str]) -> list[list[float]]:
        # Throttle to first N tokens for numerical stability and speed
        N = 96
        tokens = tokens[:N]
        # stub; replace with real hidden states later
        return [[hash(t) % 101 / 100.0] for t in tokens]

    def umap_project(self, X: list[list[float]]) -> list[tuple[float, float]]:
        # Robust UMAP projection for low-rank/degenerate inputs
        if not X:
            return []
        # Very small sequences: return simple 2D trace (no UMAP)
        if len(X) < 6:
            return [(float(i), float(row[0]) if row else 0.0) for i, row in enumerate(X)]

        try:
            import numpy as np
            import warnings as _warn

            Xnp = np.asarray(X, dtype=np.float64)
            # Ensure 2D
            if Xnp.ndim == 1:
                Xnp = Xnp.reshape(-1, 1)

            # Standardize per feature
            mu = Xnp.mean(axis=0, keepdims=True)
            sd = Xnp.std(axis=0, keepdims=True)
            Xz = (Xnp - mu) / (sd + 1e-8)

            # Replace non-finite with zeros
            if not np.isfinite(Xz).all():
                Xz = np.nan_to_num(Xz, nan=0.0, posinf=0.0, neginf=0.0)

            # If near-zero variance even after standardization, inject tiny jitter
            if float(np.var(Xz)) < 1e-10:
                Xz = Xz + np.random.normal(scale=1e-6, size=Xz.shape)

            if self._umap_available:
                import umap
                cap_in = int(self.umap_cfg.get("n_neighbors", 15))
                # safe cap: at least 2, at most len(X)-2
                n_neighbors = max(2, min(cap_in, max(2, len(Xz) - 2)))

                reducer = umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=float(self.umap_cfg.get("min_dist", 0.1)),
                    n_components=int(self.umap_cfg.get("n_components", 2)),
                    metric=str(self.umap_cfg.get("metric", "euclidean")),
                    init="random",            # avoid spectral init path entirely
                    random_state=42,
                )

                # Silence sklearn's benign matmul runtime warnings during fit
                with _warn.catch_warnings():
                    _warn.filterwarnings("ignore", category=RuntimeWarning, message=r".*matmul.*")
                    X2d = reducer.fit_transform(Xz)

                # Ensure finite outputs
                if not np.isfinite(X2d).all():
                    X2d = np.nan_to_num(X2d, nan=0.0, posinf=0.0, neginf=0.0)

                return [(float(x), float(y)) for x, y in X2d]
        except Exception:
            pass

        # Fallback: simple 2D trace
        return [(float(i), float(row[0]) if row else 0.0) for i, row in enumerate(X)]

    def betti_summary(self, X2d: list[tuple[float, float]]) -> dict:
        if self._ripser_available:
            try:
                from ripser import ripser
                import numpy as np
                pts = np.array(X2d); res = ripser(pts, maxdim=self.tda_cfg.get("max_dim", 1))
                dgms = res.get("dgms", [])
                betti0 = len(dgms[0]) if len(dgms) > 0 else 0
                betti1 = len(dgms[1]) if len(dgms) > 1 else 0
                return {"betti0": betti0, "betti1": betti1}
            except Exception:
                pass
        return {"betti0": 1, "betti1": 0}

# ───────────────────────── External verifier ──────────────────────────────────

class SimpleVerifier:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.name = model_name
        self._hf_ok = False; self._tok = None; self._enc = None
        self._device = "cpu"
        try:
            from transformers import AutoTokenizer, AutoModel  # type: ignore
            import torch  # type: ignore
            if device in ("cuda","cuda:0") and torch.cuda.is_available():
                self._device = "cuda"
            elif device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
            self._tok = AutoTokenizer.from_pretrained(self.name)
            self._enc = AutoModel.from_pretrained(self.name, low_cpu_mem_usage=True)
            self._enc.to(self._device); self._enc.eval()
            self._hf_ok = True
            print(f"[SRB][HF] Verifier using HF encoder: {self.name} (device={self._device})")
        except Exception as e:
            print(f"[SRB][HF] Verifier fallback (stub similarity): {e}")

    def _embed(self, text: str):
        if not self._hf_ok or self._tok is None or self._enc is None: return None
        try:
            import torch  # type: ignore
            with torch.no_grad():
                enc = self._tok(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
                enc = {k: v.to(self._device) for k, v in enc.items()}
                out = self._enc(**enc)
                cls = out.last_hidden_state[:, 0, :]
                cls = torch.nn.functional.normalize(cls, dim=-1)
                return cls.cpu()
        except Exception:
            return None

    def similarity(self, a: str, b: str) -> float:
        if self._hf_ok:
            try:
                import torch  # type: ignore
                ea = self._embed(a); eb = self._embed(b)
                if ea is not None and eb is not None:
                    cos = torch.nn.functional.cosine_similarity(ea, eb).item()
                    return 0.5 * (cos + 1.0)
            except Exception:
                pass
        # fallback: Jaccard
        sa = set(w.strip(".,!?;:\"'()[]{}").lower() for w in a.split() if w.strip())
        sb = set(w.strip(".,!?;:\"'()[]{}").lower() for w in b.split() if w.strip())
        if not sa and not sb: return 0.0
        inter = len(sa & sb); union = max(1, len(sa | sb))
        return inter / union

# ───────────────────────────── Runner ─────────────────────────────────────────

class Phase4Runner:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.io = IO(cfg)
        # timestamped subdir
        stamp = datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
        self.io.output_dir = self.io.output_dir / stamp
        self.io.output_dir.mkdir(parents=True, exist_ok=True)

        self.prompts = PromptBank()

        te_cfg = cfg.metrics.get("transfer_entropy", {})
        xent_cfg = cfg.metrics.get("cross_entropy", {})
        nsm_cfg = cfg.metrics.get("semantic_primitives", {})
        geo_cfg = cfg.metrics.get("geometry", {})

        self.te = TransferEntropy(**te_cfg)
        xent_device = xent_cfg.get("device", "cpu")
        ver_device  = cfg.metrics.get("verifier", {}).get("device", "cpu")
        self.xent = CrossEntropySurprise(xent_cfg.get("baseline_model", cfg.model_name), device=xent_device)
        self.nsm = SemanticPrimitiveCoherence(nsm_cfg.get("nsm_list_path", "experiments/phase3/vendors/nsm_primitives.txt"))
        self.geo = GeometryProbe(geo_cfg.get("umap", {}), geo_cfg.get("tda", {}))
        self.verifier = SimpleVerifier(cfg.verifier_model, device=ver_device)

        # Initialize HF generator for the configured model
        try:
            torch.manual_seed(cfg.seed)
        except Exception:
            pass
        try:
            _init_hf_generator(cfg.model_name, cfg.device)
        except Exception as e:
            print(f"[SRB][HF] Generator init failed (falling back to stub): {e}")

        self.capabilities = {
            "umap_used": getattr(self.geo, "_umap_available", False),
            "ripser_used": getattr(self.geo, "_ripser_available", False),
            "hf_nll_used": getattr(self.xent, "_hf_ok", False),
            "hf_verifier_used": getattr(self.verifier, "_hf_ok", False),
            "model_name": self.cfg.model_name,
            "verifier_model": self.cfg.verifier_model,
            "device_requested": self.cfg.device,
            "hf_device": _HF_DEVICE if "_HF_DEVICE" in globals() else "unset",
            "dtype": str(getattr(_HF_LM, "dtype", "unknown")) if "_HF_LM" in globals() and _HF_LM is not None else "unknown",
            "temperature": self.cfg.temperature,
        }

        # Reference store for Phase-4 stability metrics
        # keys: (bucket, run, base_prompt_text) → baseline_response
        self._baseline_ref: dict[tuple[str, int, str], str] = {}
        # previous response in the same (bucket, run) to compute traj_drift
        self._prev_resp: dict[tuple[str, int], str] = {}

    def _gen(self, prompt: str) -> str:
        # Use HF CausalLM if available; otherwise return a stub
        global _HF_LM, _HF_TOK, _HF_DEVICE
        if _HF_LM is None or _HF_TOK is None:
            return f"[MODEL OUTPUT] {prompt[:64]} …"
        try:
            enc = _HF_TOK(prompt, return_tensors="pt", truncation=True, max_length=2048)
            enc = {k: v.to(_HF_DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                out = _HF_LM.generate(
                    **enc,
                    max_new_tokens=self.cfg.max_new_tokens,
                    do_sample=True,
                    temperature=float(self.cfg.temperature),
                    top_p=float(self.cfg.top_p),
                    pad_token_id=_HF_TOK.pad_token_id or _HF_TOK.eos_token_id,
                    eos_token_id=_HF_TOK.eos_token_id,
                )
            # Slice off the prompt portion
            gen_ids = out[0][enc["input_ids"].shape[1]:]
            text = _HF_TOK.decode(gen_ids, skip_special_tokens=True)
            return text.strip()
        except Exception as e:
            return f"[MODEL OUTPUT:FALLBACK:{type(e).__name__}] {prompt[:64]} …"

    def _features_from_text(self, text: str) -> list[float]:
        alpha = sum(c.isalpha() for c in text)
        return [len(text), alpha / max(1, len(text))]

    def _ref_similarity(self, bucket: str, run_i: int, base_text: str, response: str) -> float | None:
        key = (bucket, run_i, base_text)
        ref = self._baseline_ref.get(key)
        return self.verifier.similarity(ref, response) if ref else None

    def _traj_drift(self, bucket: str, run_i: int, response: str) -> float | None:
        key = (bucket, run_i)
        prev = self._prev_resp.get(key)
        sim = self.verifier.similarity(prev, response) if prev else None
        # update for next
        self._prev_resp[key] = response
        return (1.0 - sim) if sim is not None else None

    def run(self):
        random.seed(self.cfg.seed)
        rows: list[dict] = []
        timestamp = datetime.utcnow().isoformat()

        self.io.write_json({"config": self.cfg.__dict__, "capabilities": self.capabilities, "timestamp": timestamp},
                           name="phase4_config")

        for bucket in self.cfg.buckets:
            for i in tqdm(range(self.cfg.runs), desc=f"bucket={bucket}", leave=False):
                # reset per-run previous response for trajectory drift
                self._prev_resp[(bucket, i)] = None  # type: ignore

                for inter in self.cfg.interventions:
                    base_prompt = self.prompts.sample(bucket)
                    prompt_i = Intervention.apply(inter, base_prompt.text)
                    response = self._gen(prompt_i)

                    # record baseline ref for this (bucket, run, base_prompt)
                    if inter == "baseline":
                        self._baseline_ref[(bucket, i, base_prompt.text)] = response

                    # Metrics
                    pf = self._features_from_text(prompt_i)
                    rf = self._features_from_text(response)
                    te_val = self.te.compute(pf, rf)
                    nll_val = self.xent.nll(response)
                    nsm_val = self.nsm.score(response)

                    # Geometry/TDA
                    tokens = response.split()
                    emb = self.geo.collect_embeddings(tokens)
                    traj2d = self.geo.umap_project(emb)
                    betti = self.geo.betti_summary(traj2d)

                    ver_sim = self.verifier.similarity(prompt_i, response)
                    ref_sim = self._ref_similarity(bucket, i, base_prompt.text, response)
                    traj_drift = self._traj_drift(bucket, i, response)

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
                        "ref_similarity": ref_sim,
                        "traj_drift": traj_drift,
                        "betti0": betti.get("betti0"),
                        "betti1": betti.get("betti1"),
                    }

                    rows.append(rec)
                    self.io.append_jsonl(rec, name="phase4_records")
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        try: torch.cuda.empty_cache()
                        except Exception: pass
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        try: torch.mps.empty_cache()
                        except Exception: pass

        self.io.write_csv_rows(rows, name="phase4_summary")
        print(f"Saved {len(rows)} records → {self.io.output_dir}")

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
    parser = argparse.ArgumentParser(description="SRB Phase IV Runner")
    parser.add_argument("--config", type=str, help="Path to YAML config")
    parser.add_argument("--runs", type=int)
    parser.add_argument("--model", type=str)
    parser.add_argument("--verifier", type=str)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--device", type=str, choices=["cpu","mps","cuda"])
    parser.add_argument("--output_suffix", type=str)
    args = parser.parse_args()

    # Load config
    if args.config:
        cfg_text = Path(args.config).read_text()
    else:
        default_path = Path("experiments/phase4/config/defaults.yaml")
        cfg_text = default_path.read_text() if default_path.exists() else CONFIG_YAML

    cfg = load_cfg_from_yaml(cfg_text)

    # Apply CLI overrides
    if args.runs: cfg.runs = args.runs
    if args.model: cfg.model_name = args.model
    if args.verifier: cfg.verifier_model = args.verifier
    if args.temperature is not None: cfg.temperature = args.temperature
    if args.device: cfg.device = args.device

    runner = Phase4Runner(cfg)

    if args.output_suffix:
        d = runner.io.output_dir
        runner.io.output_dir = d.with_name(d.name + f"_{args.output_suffix}")
        runner.io.output_dir.mkdir(parents=True, exist_ok=True)

    runner.run()