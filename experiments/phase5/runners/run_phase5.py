from __future__ import annotations
import argparse, json, math, os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import yaml
import contextlib
from datetime import datetime

@dataclass
class RunConfig:
    seed: int
    device: str
    dtype: str
    output_dir: Path
    model_name: str
    max_new_tokens: int
    temperature: float
    top_p: float
    embedder_name: str
    run_id: str
    prompts: List[Dict[str, Any]]
    jsonl_name: str

def load_cfg(path: str) -> RunConfig:
    cfg = yaml.safe_load(Path(path).read_text())
    return RunConfig(
        seed=cfg.get("seed", 42),
        device=cfg.get("device", "cpu"),
        dtype=cfg.get("dtype", "float16"),
        output_dir=Path(cfg["output_dir"]),
        model_name=cfg["model"]["name"],
        max_new_tokens=int(cfg["model"]["max_new_tokens"]),
        temperature=float(cfg["model"]["temperature"]),
        top_p=float(cfg["model"]["top_p"]),
        embedder_name=cfg["verifier"]["embedder"],
        run_id=cfg["experiment"]["run_id"],
        prompts=cfg["experiment"]["prompts"],
        jsonl_name=cfg["logging"]["jsonl_name"],
    )

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pick_dtype(name: str, device: str):
    name = (name or "").lower()
    if device == "mps":
        # MPS has known fp16 instability for some models/logits; prefer float32
        if name in ("float16", "fp16", "half"):
            return torch.float32
    if name in ("float16","fp16","half"):
        return torch.float16
    if name in ("bfloat16","bf16"):
        return torch.bfloat16
    return torch.float32

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return float("nan")
    return float(np.dot(a, b) / (na * nb))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    set_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_subdir = cfg.output_dir / f"{cfg.run_id}_{timestamp}"
    run_subdir.mkdir(parents=True, exist_ok=True)

    # Update records_path to use timestamped folder
    records_path = run_subdir / cfg.jsonl_name

    # Save config snapshot for provenance (top-level + inside run folder)
    cfg_snapshot = {
        "seed": cfg.seed,
        "device": cfg.device,
        "dtype": cfg.dtype,
        "output_dir": str(cfg.output_dir),
        "model": {
            "name": cfg.model_name,
            "max_new_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
        },
        "verifier": {"embedder": cfg.embedder_name},
        "experiment": {
            "run_id": cfg.run_id,
            "num_prompts": len(cfg.prompts),
            "jsonl_name": cfg.jsonl_name,
            "timestamp": timestamp,
        },
    }
    # Write alongside output root with a timestamped name
    (cfg.output_dir / f"{cfg.run_id}_{timestamp}_config.yml").write_text(
        yaml.safe_dump(cfg_snapshot, sort_keys=False)
    )
    # Also write inside the run directory for co-location with artifacts
    (run_subdir / "config.yml").write_text(
        yaml.safe_dump(cfg_snapshot, sort_keys=False)
    )

    device = cfg.device
    dtype = pick_dtype(cfg.dtype, device)

    # Optionally, ensure default dtype for MPS stability
    if device == "mps" and dtype == torch.float32:
        torch.set_default_dtype(torch.float32)

    # Model + tokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=dtype,
        device_map=None,
    )
    model.to(device)
    model.eval()
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.eos_token_id

    # Small env fix for tokenizer parallelism warnings
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Embedder
    embedder = SentenceTransformer(cfg.embedder_name)

    # Helper: compute per-step logprobs & text stream
    def generate_stepwise(prompt: str):
        """
        Decoding with output scores so we can compute per-token logprobs.
        Includes guards against NaN/Inf probability issues on some backends by
        retrying with greedy decoding.
        """
        inputs = tok(prompt, return_tensors="pt").to(device)

        def _run(do_sample: bool):
            return model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=cfg.temperature if do_sample else None,
                top_p=cfg.top_p if do_sample else None,
                max_new_tokens=cfg.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tok.eos_token_id,
            )

        with torch.inference_mode():
            try:
                gen = _run(do_sample=(cfg.temperature > 0.0))
            except RuntimeError as e:
                msg = str(e).lower()
                if "probability tensor contains either" in msg or "nan" in msg or "inf" in msg:
                    # Retry deterministically without sampling
                    gen = _run(do_sample=False)
                else:
                    raise

        seq = gen.sequences[0]  # (prompt + gen)
        new_ids = seq[len(inputs["input_ids"][0]):]  # only the new part
        scores = gen.scores  # list[tensor: (batch, vocab)] length == new_tokens

        cum_texts = []
        tokens = []
        logprobs = []
        surprisal = []

        cur_ids = inputs["input_ids"][0].clone()
        for t, (tid, logits) in enumerate(zip(new_ids, scores), start=1):
            # Defensive clamp to avoid NaNs in log_softmax on odd backends
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            logits = torch.clamp(logits, -1e4, 1e4)

            lp_vec = torch.log_softmax(logits[0], dim=-1)
            lp = lp_vec[tid]
            logprobs.append(float(lp))
            surprisal.append(float(-lp))  # -log p

            cur_ids = torch.cat([cur_ids, tid.view(1)], dim=0)
            cum_texts.append(tok.decode(cur_ids, skip_special_tokens=True))
            tok_str = tok.decode(tid.view(1), skip_special_tokens=True)
            if not tok_str:
                tok_str = tok.convert_ids_to_tokens(int(tid))
            tokens.append(tok_str)

        return tokens, new_ids.tolist(), logprobs, surprisal, cum_texts

    # Logging loop
    with records_path.open("w") as f:
        for p in cfg.prompts:
            prompt_id = p["id"]
            bucket = p.get("bucket", "")
            prompt_text = p["text"].strip()

            tokens, ids, lps, sps, cum_texts = generate_stepwise(prompt_text)

            # Precompute prompt embedding once
            prompt_emb = embedder.encode([prompt_text], convert_to_numpy=True, normalize_embeddings=True)[0]
            prev_emb = None

            for t in tqdm(range(len(tokens)), desc=f"prompt={prompt_id}"):
                cum = cum_texts[t]
                cur_emb = embedder.encode([cum], convert_to_numpy=True, normalize_embeddings=True)[0]

                cos_to_prev = cosine(cur_emb, prev_emb) if prev_emb is not None else np.nan
                cos_to_prompt = cosine(cur_emb, prompt_emb)

                rec = {
                    "type": "phase5_step",
                    "run_id": cfg.run_id,
                    "model_name": cfg.model_name,
                    "prompt_id": prompt_id,
                    "prompt_text": prompt_text,
                    "bucket": bucket,
                    "t": t + 1,
                    "token": tokens[t],
                    "token_id": int(ids[t]),
                    "logprob": lps[t],
                    "surprisal": sps[t],
                    "cos_to_prev": cos_to_prev,
                    "cos_to_prompt": cos_to_prompt,
                    "cum_text": cum,
                }
                f.write(json.dumps(rec) + "\n")
                prev_emb = cur_emb

    print(f"[phase5] Wrote stepwise records → {records_path}\n[phase5] Config snapshot → {(run_subdir / 'config.yml')}\n[phase5] Run dir → {run_subdir}")

if __name__ == "__main__":
    main()