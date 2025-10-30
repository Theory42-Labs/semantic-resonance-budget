"""
SRB Generation Loops
--------------------
Implements baseline and SRB-dynamic text generation loops using PyTorch + Transformers.
These loops rely on `entropy_utils` for entropy computation and plateau detection.

New helper functions are added for chat formatting, stop phrase detection, and list counting guards.
Dynamic sampling uses top-p sampling and includes stop conditions for EOS token, stop-phrases,
list count guard, and entropy plateau to improve SRB drift detection and stopping criteria.
"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional
import re
import os
import math
import numpy as np

# --- SRB defaults & constants (must be defined before use) ---
DEFAULT_TOPK = 20

# --- Device resolver ---
def _resolve_device(device: Optional[str] = "mps") -> str:
    """
    Resolve a user-supplied device string into a valid torch device.
    Priority: explicit != 'auto' > CUDA > MPS (Apple Silicon) > CPU.
    Returns one of: 'cuda', 'mps', or 'cpu'.
    """
    try:
        if device and device != "auto":
            return str(device)
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"
def _get_last_token_embedding(outputs, layer: int = -1):
    """
    Return the embedding vector (np.ndarray) for the last generated token
    from the specified hidden state layer.
    """
    if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
        return None
    hs = outputs.hidden_states[layer]  # [batch, seq, dim]
    vec = hs[0, -1, :].detach().cpu().numpy()
    return vec

def _cosine_sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 1.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 1.0
    return float(np.dot(a, b) / (na * nb))

def _normalize_entropy(H: float, k: int = DEFAULT_TOPK) -> float:
    # Normalize Shannon entropy by log(k) so values land in [0,1].
    denom = max(1e-12, math.log(k))
    x = max(0.0, min(1.0, H / denom))
    return x

from .utils import topk_entropy_from_logits, entropy_plateau, should_soft_stop

# ---------------------------
# Helper functions for chat formatting and SRB drift guards
# ---------------------------

def build_chat_prompt(tokenizer, prompt: str) -> str:
    """Render a chat-formatted prompt using the tokenizer's chat template when available.
    Returns a plain string; the caller is responsible for tokenization.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            # Older transformers may require positional arg name `conversation`
            return tokenizer.apply_chat_template(
                messages,
                False,  # tokenize
                add_generation_prompt=True,
            )
    # Fallback manual template
    return f"<|system|>You are a helpful assistant.<|end|><|user|>{prompt}<|end|><|assistant|>"

STOP_PHRASES = [
    "Thank you for using our service.",
    "If you have any more questions, feel free to ask.",
    "Let me know if there's anything else I can help with.",
]

def contains_stop_phrase(text: str) -> bool:
    """
    Return True if any of the defined stop phrases appear in the given text.
    """
    text_lower = text.lower()
    return any(phrase.lower() in text_lower for phrase in STOP_PHRASES)

def count_required_items_from_prompt(prompt: str) -> int:
    """
    Detect a requested item count from the prompt. Handles numerals (e.g., 5),
    number words (e.g., five), and patterns like "(max 6 steps)" across common domains.
    Returns 0 if no explicit count is found.
    """
    text = prompt.lower()

    # 1) (max N ...)
    m = re.search(r"\(\s*max\s*(\d{1,2})\s*([a-z]+)?\s*\)", text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass

    # 2) N <domain>
    DOMAINS = r"islands|steps|questions|answers|items|points|tips|facts"
    m = re.search(rf"\b(\d{{1,2}})\s+(?:major\s+|practical\s+|concise\s+)?(?:[a-z]+\s+)?(?:{DOMAINS})\b", text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass

    # 3) number-word <domain>
    WORD2NUM = {
        'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,
        'eleven':11,'twelve':12
    }
    m = re.search(rf"\b({'|'.join(WORD2NUM.keys())})\b\s+(?:major\s+|practical\s+|concise\s+)?(?:[a-z]+\s+)?(?:{DOMAINS})\b", text)
    if m:
        return WORD2NUM.get(m.group(1), 0)

    # 4) verbs + count: list|give|provide|write|name followed by count
    m = re.search(rf"\b(list|give|provide|write|name)\b[^\n\r]{{0,40}}\b(\d{{1,2}})\b[^\n\r]{{0,20}}\b(?:{DOMAINS})\b", text)
    if m:
        try:
            return int(m.group(2))
        except Exception:
            pass

    # 5) FAQ: "3 questions and answers"
    m = re.search(r"\b(\d{1,2})\s+questions?\s+and\s+answers?\b", text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass

    return 0

def current_item_count(text: str) -> int:
    """Count list/FAQ items; require visible content after the marker."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    n_ordered = 0
    for ln in lines:
        if re.match(r"^\s*(\d+)[\.)]\s+\S", ln):
            n_ordered += 1
    n_bullets = 0
    for ln in lines:
        if re.match(r"^\s*[-*•–—]\s+\S", ln):
            n_bullets += 1
    n_q = 0
    for ln in lines:
        if re.match(r"^\s*\*?\*?\s*Q\s*\d+\s*[:\-]\s*\S", ln, re.IGNORECASE):
            n_q += 1
    return max(n_ordered, n_bullets, n_q)

# Helper: detect if a list is in progress in the last few lines
def list_in_progress(text: str) -> bool:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    for ln in lines[-5:]:  # look at recent lines
        if re.match(r"^\s*(\d+)[\.)]\s+", ln) or re.match(r"^\s*[-*•–—]\s+", ln) or re.match(r"^\s*\*?\*?\s*Q\s*\d+\s*[:\-]", ln, re.IGNORECASE):
            return True
    return False

# --- New helpers for stricter list/FAQ item completeness and EOS soft-stop detection ---
def last_item_complete(text: str) -> bool:
    """Heuristic: last list/FAQ item has substance and ends with sentence punctuation."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    start_ix = None
    for i in range(len(lines) - 1, -1, -1):
        ln = lines[i]
        if re.match(r"^\s*(\d+)[\.)]\s+\S", ln) or \
           re.match(r"^\s*[-*•–—]\s+\S", ln) or \
           re.match(r"^\s*\*?\*?\s*Q\s*\d+\s*[:\-]\s*\S", ln, re.IGNORECASE):
            start_ix = i
            break
    if start_ix is None:
        return False
    block_lines = []
    for j in range(start_ix, len(lines)):
        ln = lines[j]
        if j > start_ix and (
            re.match(r"^\s*(\d+)[\.)]\s+\S", ln) or 
            re.match(r"^\s*[-*•–—]\s+\S", ln) or 
            re.match(r"^\s*\*?\*?\s*Q\s*\d+\s*[:\-]\s*\S", ln, re.IGNORECASE)
        ):
            break
        block_lines.append(ln)
    block = "\n".join(block_lines).strip()
    if len(re.sub(r"\s+", "", block)) < 20:
        return False
    return block.endswith((".", "!", "?", "…"))

def _soft_done_strict(text: str) -> bool:
    t = (text or "").rstrip()
    if not t:
        return False
    if t.endswith(":"):
        return False
    return t.endswith((".", "!", "?", "…"))

# ---------------------------
# Config defaults
# ---------------------------
DEFAULT_WINDOW = 16
DEFAULT_SLOPE_EPS = 0.01
DEFAULT_MIN_TOKENS = 24
DEFAULT_STATIC_MAX_TOKENS = 256
DEFAULT_TOP_P = 0.9

# ---------------------------
# Utility: run one generation pass (static baseline)
# ---------------------------
def run_static(model, tokenizer, device: str, prompt: str, max_new_tokens: int = DEFAULT_STATIC_MAX_TOKENS) -> Dict:
    """
    Run a static generation loop with sampling and fixed max tokens.
    Uses chat formatting and trims output at EOS token if present.
    """
    messages = build_chat_prompt(tokenizer, prompt)
    enc = tokenizer(messages, return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=DEFAULT_TOPK,
            top_p=DEFAULT_TOP_P,
            eos_token_id=tokenizer.eos_token_id,
        )
    t1 = time.time()

    # Extract only the generated continuation tokens
    gen_ids = out[0][enc["input_ids"].shape[1]:]

    # Trim at EOS token if present and record stop_reason accordingly
    stop_reason = "max_tokens"
    if tokenizer.eos_token_id is not None:
        eos_positions = (gen_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            gen_ids = gen_ids[:eos_positions[0]]
            stop_reason = "eos_token"

    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # --- Per-token entropy & p_eos logging by replaying logits stepwise ---
    entropies = []
    steps = []
    eos_id = tokenizer.eos_token_id
    with torch.no_grad():
        o = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=True, output_hidden_states=True)
        past = o.past_key_values
        # Iterate over the *emitted* tokens and replay one by one
        for i, tid in enumerate(gen_ids):
            # logits predicting this tid (based on the current prefix)
            logits = o.logits[0, -1, :]
            # entropy over the next-token distribution (Shannon)
            H = float(topk_entropy_from_logits(logits, k=DEFAULT_TOPK))
            entropies.append(H)
            # probability mass on EOS at this point (no temperature applied for logging)
            probs = torch.softmax(logits, dim=-1)
            p_eos = float(probs[eos_id].item()) if eos_id is not None else 0.0
            # record step with the actual sampled token id
            steps.append({
                "step": int(i + 1),
                "token_id": int(tid.item() if hasattr(tid, "item") else int(tid)),
                "entropy": H,
                "p_eos": p_eos,
            })
            # advance with the observed token
            o = model(input_ids=tid.view(1, 1).to(device), use_cache=True, past_key_values=past, output_hidden_states=True)
            past = o.past_key_values

    avgH = float(sum(entropies) / max(1, len(entropies)))
    finalH = float(entropies[-1]) if entropies else float('nan')

    return dict(
        strategy="static",
        tokens=int(len(gen_ids)),
        wall_time=float(t1 - t0),
        avg_entropy=avgH,
        final_entropy=finalH,
        stop_reason=stop_reason,
        text=text,
        entropies=entropies,
        steps=steps,
    )


def run_srb_dynamic(model, tokenizer, prompt: str, cfg: Dict) -> Dict:
    """
    Run a dynamic SRB generation loop with top-p sampling and multiple stopping criteria:
    - Entropy plateau detection combined with soft stop phrases.
    - EOS token detection.
    - Stop phrases detection.
    - List count guard based on prompt instructions.
    This improves stopping behavior and guards against SRB drift.

    Returns:
        dict with keys including:
          - 'text', 'tokens', 'wall_time', 'avg_entropy', 'final_entropy', 'stop_reason', 'entropies'
          - 'steps': per-token logs [{'step','token','entropy','p_eos','plateau','soft_done','slope','score','text_len'}, ...]
          - 'srb_budget': scalar Σ |slope| * H across steps (proxy for semantic energy spent)
    """
    device = cfg.get("device", "cpu")
    min_tokens = int(cfg.get("min_tokens", DEFAULT_MIN_TOKENS))
    max_tokens = int(cfg.get("max_tokens", DEFAULT_STATIC_MAX_TOKENS))
    window = int(cfg.get("completion_window", DEFAULT_WINDOW))
    slope_eps = float(cfg.get("completion_eps", DEFAULT_SLOPE_EPS))
    temperature = float(cfg.get("temperature_base", 0.8))
    top_p = float(cfg.get("top_p_base", DEFAULT_TOP_P))
    entropy_k = int(cfg.get("entropy_top_k", DEFAULT_TOPK))
    # --- Intervention config ---
    intervention = str(cfg.get("intervention", "none")).lower()

    messages = build_chat_prompt(tokenizer, prompt)
    enc = tokenizer(messages, return_tensors="pt").to(device)

    required_count = count_required_items_from_prompt(prompt)

    seed = int(os.environ.get("SRB_SEED", 1234))
    no_guards = os.environ.get("SRB_NO_GUARDS", "0") == "1"
    # EOS-aware SRB parameters
    EOS_ARM = float(os.environ.get("SRB_EOS_ARM", 0.20))
    EOS_FIRE = float(os.environ.get("SRB_EOS_FIRE", 0.50))
    EOS_PATIENCE = int(os.environ.get("SRB_EOS_PATIENCE", 8))
    EOS_TAU = float(os.environ.get("SRB_EOS_TAU", 0.05))
    EOS_WEIGHT = float(os.environ.get("SRB_EOS_WEIGHT", 0.8))
    SRB_SCORE_THRESH = float(os.environ.get("SRB_SCORE_THRESH", 2.5))
    torch.manual_seed(seed)

    # --- SRB research logging ---
    per_step_data: List[Dict] = []
    srb_budget: float = 0.0  # cumulative semantic "energy" spent ~ Σ |slope| * H

    with torch.no_grad():
        o = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=True, output_hidden_states=True)
        past = o.past_key_values
        emitted_ids = []
        entropies: List[float] = []
        coherences: List[float] = []
        resonances: List[float] = []
        prev_embed = None
        prev_R = None
        cur_text = ""
        stop_reason = "unknown"
        t0 = time.time()
        eos_armed = False
        eos_since = 0
        ringbuf: List[Dict] = []

        while True:
            # Full softmax with temperature for EOS probability
            logits = o.logits[0, -1, :]
            full_probs = torch.softmax(logits / max(1e-6, temperature), dim=-1)
            p_eos = float(full_probs[tokenizer.eos_token_id]) if tokenizer.eos_token_id is not None else 0.0
            H = topk_entropy_from_logits(logits, k=entropy_k)
            entropies.append(H)

            # Default indicators for this step (filled after min_tokens gate)
            plateau = False
            soft_done = False
            slope = 0.0
            score = float("nan")

            # Nucleus (top-p) sampling using full_probs
            sorted_probs, sorted_idx = torch.sort(full_probs, descending=True)
            cdf = torch.cumsum(sorted_probs, dim=-1)
            cutoff_idx = int(torch.searchsorted(cdf, torch.tensor(top_p, device=full_probs.device)))
            cutoff_idx = max(1, cutoff_idx)
            cand_idx = sorted_idx[:cutoff_idx]
            cand_probs = full_probs[cand_idx]
            perm_idx = cand_idx[torch.randperm(cand_idx.numel())] if intervention == "logit_shuffle" else cand_idx
            pick_rel = torch.multinomial(cand_probs, num_samples=1)
            if intervention == "logit_shuffle":
                next_token_id = int(perm_idx[pick_rel].item())
            else:
                next_token_id = int(cand_idx[pick_rel].item())

            emitted_ids.append(next_token_id)

            o = model(input_ids=torch.tensor([[next_token_id]], device=device), use_cache=True, past_key_values=past, output_hidden_states=True)
            past = o.past_key_values
            cur_text = tokenizer.decode(torch.tensor(emitted_ids), skip_special_tokens=True)

            # --- Semantic coherence and resonance computation ---
            Hn = _normalize_entropy(H, k=entropy_k)
            # Intervention for embedding noise
            hvec = o.hidden_states[-1][0, -1, :].detach().cpu()
            if intervention == "embed_noise":
                dim = hvec.shape[-1]
                std = float(hvec.std().item()) or 1.0
                embed_t = np.random.normal(0.0, std, size=(dim,))
            else:
                embed_t = _get_last_token_embedding(o, layer=-1)
            C_t = _cosine_sim(embed_t, prev_embed)
            R_t = max(0.0, (1.0 - Hn) * C_t)
            dR_t = 0.0 if prev_R is None else (R_t - prev_R)
            coherences.append(C_t)
            resonances.append(R_t)
            prev_embed = embed_t
            prev_R = R_t

            # Accumulate SRB budget (~ semantic change magnitude * uncertainty)
            srb_budget += abs(slope) * H

            # Check stop conditions
            if tokenizer.eos_token_id is not None and next_token_id == tokenizer.eos_token_id:
                stop_reason = "eos_token"
                # Per-step logging (before break)
                try:
                    per_step_data.append({
                        "step": len(emitted_ids),
                        "token": next_token_id,
                        "entropy": H,
                        "p_eos": p_eos,
                        "plateau": bool(plateau),
                        "soft_done": bool(soft_done),
                        "slope": float(slope),
                        "score": float(score) if not math.isnan(score) else None,
                        "text_len": len(cur_text),
                        "text": cur_text,
                        "temperature": float(temperature),
                        "top_p": float(top_p),
                        "coherence": float(C_t),
                        "resonance": float(R_t),
                        "d_resonance": float(dR_t),
                        "intervention": intervention,
                    })
                except Exception:
                    pass
                break

            if not no_guards and contains_stop_phrase(cur_text):
                stop_reason = "stop_phrase_detected"
                try:
                    per_step_data.append({
                        "step": len(emitted_ids),
                        "token": next_token_id,
                        "entropy": H,
                        "p_eos": p_eos,
                        "plateau": bool(plateau),
                        "soft_done": bool(soft_done),
                        "slope": float(slope),
                        "score": float(score) if not math.isnan(score) else None,
                        "text_len": len(cur_text),
                        "text": cur_text,
                        "temperature": float(temperature),
                        "top_p": float(top_p),
                        "coherence": float(C_t),
                        "resonance": float(R_t),
                        "d_resonance": float(dR_t),
                        "intervention": intervention,
                    })
                except Exception:
                    pass
                break

            # Remove explicit list_count_guard here (block removed)

            if len(emitted_ids) >= min_tokens:
                # Use utils.entropy_plateau boolean API; derive slope & mean locally
                plateau = entropy_plateau(entropies, window, slope_eps)
                slope = (entropies[-1] - entropies[-2]) if len(entropies) >= 2 else 0.0
                meanH = sum(entropies[-window:]) / float(min(window, len(entropies))) if entropies else 0.0
                soft_done = _soft_done_strict(cur_text)

                item_count = current_item_count(cur_text)
                in_list = list_in_progress(cur_text)
                last_ok = last_item_complete(cur_text)

                # Structure readiness: if a list seems in progress, require the last item to be complete.
                structure_ready = (not in_list) or last_ok or no_guards

                # EOS state machine
                if p_eos >= EOS_ARM and not eos_armed:
                    eos_armed, eos_since = True, 0
                elif eos_armed and p_eos < EOS_ARM:
                    eos_since += 1

                # EOS-weighted SRB score
                eos_bonus = 0.0
                if eos_armed:
                    z = (p_eos - EOS_FIRE) / max(1e-6, EOS_TAU)
                    eos_bonus = 1.0 / (1.0 + math.exp(-z))
                score = (1.0 if plateau else 0.0) + (1.0 if soft_done else 0.0) + EOS_WEIGHT * eos_bonus


                # Primary stop decisions
                if structure_ready and ((eos_armed and p_eos >= EOS_FIRE and soft_done) or (score >= SRB_SCORE_THRESH)):
                    stop_reason = f"eos+entropy_stop (p_eos={p_eos:.2f}, slope={slope:.4f}, score={score:.2f})"
                    try:
                        per_step_data.append({
                            "step": len(emitted_ids),
                            "token": next_token_id,
                            "entropy": H,
                            "p_eos": p_eos,
                            "plateau": bool(plateau),
                            "soft_done": bool(soft_done),
                            "slope": float(slope),
                            "score": float(score) if not math.isnan(score) else None,
                            "text_len": len(cur_text),
                            "text": cur_text,
                            "temperature": float(temperature),
                            "top_p": float(top_p),
                            "coherence": float(C_t),
                            "resonance": float(R_t),
                            "d_resonance": float(dR_t),
                            "intervention": intervention,
                        })
                    except Exception:
                        pass
                    break

                # Track candidate checkpoints for fallback
                ringbuf.append({
                    "pos": len(emitted_ids),
                    "punct": soft_done,
                    "p_eos": p_eos,
                    "abs_slope": abs(slope),
                    "last_ok": last_ok,
                })
                if len(ringbuf) > 16:
                    ringbuf.pop(0)

                # Patience fallback: if EOS armed but not converging, pick best recent stop
                if eos_armed and eos_since >= EOS_PATIENCE:
                    best_pos = None
                    best_score = -1e9
                    for r in ringbuf:
                        s = (2.0 if r["last_ok"] else 0.0) + (1.0 if r["punct"] else 0.0) + 0.8 * r["p_eos"] - 0.5 * r["abs_slope"] - 0.01 * r["pos"]
                        if s > best_score:
                            best_score, best_pos = s, r["pos"]
                    if best_pos is not None and best_pos < len(emitted_ids):
                        emitted_ids = emitted_ids[:best_pos]
                        cur_text = tokenizer.decode(torch.tensor(emitted_ids), skip_special_tokens=True)
                    stop_reason = "eos_patience_fallback"
                    try:
                        per_step_data.append({
                            "step": len(emitted_ids),
                            "token": next_token_id,
                            "entropy": H,
                            "p_eos": p_eos,
                            "plateau": bool(plateau),
                            "soft_done": bool(soft_done),
                            "slope": float(slope),
                            "score": float(score) if not math.isnan(score) else None,
                            "text_len": len(cur_text),
                            "text": cur_text,
                            "temperature": float(temperature),
                            "top_p": float(top_p),
                            "coherence": float(C_t),
                            "resonance": float(R_t),
                            "d_resonance": float(dR_t),
                            "intervention": intervention,
                        })
                    except Exception:
                        pass
                    break

            # Log per-step metrics for research/visualization
            try:
                per_step_data.append({
                    "step": len(emitted_ids),
                    "token": next_token_id,
                    "entropy": H,
                    "p_eos": p_eos,
                    "plateau": bool(plateau),
                    "soft_done": bool(soft_done),
                    "slope": float(slope),
                    "score": float(score) if not math.isnan(score) else None,
                    "text_len": len(cur_text),
                    "text": cur_text,
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "coherence": float(C_t),
                    "resonance": float(R_t),
                    "d_resonance": float(dR_t),
                    "intervention": intervention,
                })
            except Exception:
                # Never fail generation due to logging
                pass

            if len(emitted_ids) >= max_tokens:
                stop_reason = "cap_safety"
                break

        t1 = time.time()

    avgH = float(sum(entropies) / max(1, len(entropies)))
    finalH = float(entropies[-1]) if entropies else float('nan')
    avgC = float(sum(coherences) / max(1, len(coherences)))
    finalC = float(coherences[-1]) if coherences else float('nan')
    avgR = float(sum(resonances) / max(1, len(resonances)))
    finalR = float(resonances[-1]) if resonances else float('nan')

    return dict(
        strategy="srb_dynamic",
        tokens=len(emitted_ids),
        wall_time=t1 - t0,
        avg_entropy=avgH,
        final_entropy=finalH,
        avg_coherence=avgC,
        final_coherence=finalC,
        avg_resonance=avgR,
        final_resonance=finalR,
        stop_reason=stop_reason,
        text=cur_text,
        entropies=entropies,
        coherences=coherences,
        resonances=resonances,
        srb_budget=srb_budget,
        per_step_data=per_step_data,
        steps=per_step_data,
        intervention=intervention,
    )


# ---------------------------
# Helper to load model/tokenizer
# ---------------------------
def load_model(model_id: str, device: Optional[str] = "mps"):
    device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()
    try:
        model.config.output_hidden_states = True
    except Exception:
        pass
    return model, tokenizer, device