#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_length.py — Rolling-context runner for BoolQ-length evals.

This preserves TRUE multi-turn behavior:
T1 → A1 → T2 → A2 → … → TL → AL, with each assistant reply appended to the context.

What’s included:
- Input is boolq_final.jsonl.
- Scaffolds: baseline, meta, semantic, underspecified, misleading.
- Response types: tri (YES/NO/IDK) or bi (YES/NO only)
- Strict template filenames:
    llm_length_study/data/length/{scaffold}/L{L}_{scaffold}_{response_type}.jsonl
  And for 'misleading' we branch by gold:
    answer==true  -> llm_length_study/data/length/misleading/L{L}_misleading_true_{response_type}.jsonl
    answer==false -> llm_length_study/data/length/misleading/L{L}_misleading_false_{response_type}.jsonl
  And for 'baseline':
    llm_length_study/data/length/baseline/L1_baseline_{response_type}.jsonl
- Deterministic anchor generator for {anchor_a..d} (underspecified, misleading).
- Progress prints so you can see where it’s working/lagging.
- Model aliases: phi4-mini, qwen2.5, deepseek7b (or pass full HF model id).
- Optional --dry-run writes prompts/transcripts without importing torch/transformers.

Examples:
# Tri-nary responses (YES/NO/IDK) - default
python llm_length_study/run_length.py \
  --scaffold meta \
  --lengths 6,11,16,21 \
  --in-final llm_length_study/data/boolq_final.jsonl \
  --out-root runs/qwen2.5_meta_tri \
  --model qwen2.5 \
  --device cuda \
  --response-type tri \
  --skip-existing

# Binary responses (YES/NO only)
python llm_length_study/run_length.py \
  --scaffold meta \
  --lengths 6,11,16,21 \
  --in-final llm_length_study/data/boolq_final.jsonl \
  --out-root runs/qwen2.5_meta_bi \
  --model qwen2.5 \
  --device cuda \
  --response-type bi \
  --skip-existing

Test specific questions:
python llm_length_study/run_length.py \
  --scaffold misleading \
  --lengths 6,11,16,21 \
  --in-final llm_length_study/data/boolq_final.jsonl \
  --out-root runs/qwen2.5_misleading_test \
  --model qwen2.5 \
  --device cuda \
  --response-type tri \
  --id-include dev_0023,dev_0039,dev_0055

# Reasoning models (DeepSeek-R1-Distill)
python llm_length_study/run_length.py \
  --scaffold semantic \
  --lengths 6,11,16,21 \
  --in-final llm_length_study/data/boolq_final.jsonl \
  --out-root runs/deepseek_r1_semantic \
  --model unsloth/DeepSeek-R1-Distill-Llama-8B \
  --device cuda \
  --response-type tri \
  --max-new-tokens-tasks 200 \
  --max-new-tokens-final 800 \
  --extract-reasoning \
  --load-in-4bit \
  --skip-existing

# Reasoning models (Qwen3 with thinking mode)
python llm_length_study/run_length.py \
  --scaffold semantic \
  --lengths 6,11,16,21 \
  --in-final llm_length_study/data/boolq_final.jsonl \
  --out-root runs/qwen3_thinking_semantic \
  --model Qwen/Qwen3-8B \
  --device cuda \
  --response-type tri \
  --max-new-tokens-tasks 200 \
  --max-new-tokens-final 800 \
  --enable-thinking \
  --extract-reasoning \
  --load-in-4bit \
  --skip-existing
"""

from __future__ import annotations
import argparse
import json
import re
import time
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Iterable, Any, Optional

# --------------------------------------------------------------------------------------
# Paths & constants
# --------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
LENGTH_DIR   = DATA_DIR / "length"

SCAFFOLDS = ("baseline", "meta", "semantic", "underspecified", "misleading")

# Convenience aliases; pass full HF model id to --model to bypass
MODEL_ALIASES = {
    "phi4-mini": "microsoft/Phi-4-mini-instruct",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "deepseek7b": "deepseek-ai/deepseek-llm-7b-chat",
}

# --------------------------------------------------------------------------------------
# Light IO helpers
# --------------------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON at {path}:{ln} -> {e}")
    return rows

def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

# --------------------------------------------------------------------------------------
# Template plumbing
# --------------------------------------------------------------------------------------
PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z0-9_]+)\}")

def find_placeholders(s: str) -> List[str]:
    return PLACEHOLDER_RE.findall(s)

def strict_format(template: str, mapping: Dict[str, Any], origin: str) -> str:
    needed = set(find_placeholders(template))
    missing = [k for k in needed if k not in mapping]
    if missing:
        raise KeyError(f"Missing placeholders in {origin}: {missing}")
    try:
        return template.format(**mapping)
    except Exception as e:
        raise RuntimeError(f"Failed formatting {origin}: {e}")

def resolve_template_path(scaffold: str, L: int, gold_bool: bool, response_type: str = "tri") -> Path:
    """
    Return the exact template file path for (scaffold, L, gold, response_type).
    response_type: 'tri' (yes/no/idk) or 'bi' (yes/no only)
    Strict names only; no fallbacks.
    """
    base = LENGTH_DIR / scaffold
    if scaffold == "misleading":
        base_name = "misleading_true" if gold_bool else "misleading_false"
        fname = f"L{L}_{base_name}_{response_type}.jsonl"
        path = base / fname
    elif scaffold == "baseline":
        path = base / f"L1_baseline_{response_type}.jsonl"
    else:
        # meta | semantic | underspecified
        path = base / f"L{L}_{scaffold}_{response_type}.jsonl"

    if not path.exists():
        raise FileNotFoundError(f"[TPL] Missing template: {path}")
    return path

# --------------------------------------------------------------------------------------
# Anchor generator (deterministic; used for underspecified & misleading)
# --------------------------------------------------------------------------------------
def _seed_from_topic_or_question(topic_primary: Optional[str], question: str) -> str:
    topic = (topic_primary or "").strip()
    if topic:
        return topic
    toks = re.findall(r"[A-Za-z0-9]+", question.lower())
    return " ".join(toks[:2]) if toks else "topic"

def _title_head(seed: str) -> str:
    words = seed.strip().split()
    if not words:
        return "Topic"
    return " ".join(w[:1].upper() + w[1:] for w in words[:2])

def _acronym(seed: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z]+", seed)]
    if not words:
        return "SYS"
    if len(words) == 1:
        w = words[0]
        return (w[:4] if len(w) >= 4 else w).upper()
    return "".join(w[0].upper() for w in words)[:4]

_ANCHOR_PATTERNS = [
    ("{head} Reference Council", "{acr}RC"),
    ("{head} Standards Council", "{acr}SC"),
    ("{head} Methods Initiative", "{acr}MI"),
    ("{head} Labeling Guide", "{acr}LG"),
]

def generate_anchors(topic_primary: Optional[str], question: str) -> Dict[str, str]:
    seed = _seed_from_topic_or_question(topic_primary, question)
    head = _title_head(seed)
    acr  = _acronym(seed)
    h = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16)
    order = [(h + i) % len(_ANCHOR_PATTERNS) for i in range(len(_ANCHOR_PATTERNS))]
    names = ["anchor_a", "anchor_b", "anchor_c", "anchor_d"]
    anchors = {}
    for i, name in enumerate(names):
        pi = order[i % len(_ANCHOR_PATTERNS)]
        tpat, apat = _ANCHOR_PATTERNS[pi]
        title = tpat.format(head=head, acr=acr)
        acro  = apat.format(head=head, acr=acr)
        anchors[name] = f"{title} ({acro})"
    return anchors

# --------------------------------------------------------------------------------------
# Placeholder mapping per scaffold
# --------------------------------------------------------------------------------------
def build_placeholder_map(item: dict, scaffold: str) -> Dict[str, Any]:
    question = (item.get("corrected") or item.get("question") or "").strip()
    topic_primary = (item.get("topic_primary") or "").strip()
    topic_related = item.get("topic_related") or []

    mapping: Dict[str, Any] = {
        "QUESTION": question,
        "topic_primary": topic_primary if topic_primary else _seed_from_topic_or_question(topic_primary, question),
        "GOLD": "YES" if bool(item.get("answer")) else "NO",
    }

    if scaffold == "semantic":
        # pad related terms to 4
        rel = [str(x) for x in topic_related]
        rel += ["", "", "", ""]
        mapping.update({
            "related_a": rel[0],
            "related_b": rel[1],
            "related_c": rel[2],
            "related_d": rel[3],
        })

    if scaffold in ("underspecified", "misleading"):
        mapping.update(generate_anchors(topic_primary, question))

    return mapping

# --------------------------------------------------------------------------------------
# Lazy HF import (used ONLY when not --dry-run)
# --------------------------------------------------------------------------------------
def _lazy_hf():
    import torch  # imported only if actually generating
    from transformers import AutoTokenizer, AutoModelForCausalLM
    return torch, AutoTokenizer, AutoModelForCausalLM

# --------------------------------------------------------------------------------------
# Reasoning model support
# --------------------------------------------------------------------------------------
def extract_answer_from_reasoning(reply: str, model_name: str = "") -> str:
    """
    Extract final answer from reasoning model output.
    
    Handles multiple reasoning tag formats:
    - <think>...</think> (DeepSeek-R1, Qwen3)
    - <reasoning>...</reasoning>
    - [Thinking]...[/Thinking]
    - Plain text reasoning (fallback: take last line)
    
    Returns the answer portion only (everything after closing tag or original if no tags)
    """
    # Common reasoning tag patterns (ordered by specificity)
    tag_patterns = [
        (r'</think>', r'<think>'),           # DeepSeek-R1, Qwen3
        (r'</reasoning>', r'<reasoning>'),   # Alternative format
        (r'\[/Thinking\]', r'\[Thinking\]'), # Bracket style
        (r'</thought>', r'<thought>'),       # Another common format
    ]
    
    # Try each tag pattern
    for close_tag, open_tag in tag_patterns:
        if close_tag in reply:
            # Split on closing tag and take everything after
            parts = reply.split(close_tag)
            if len(parts) > 1:
                answer = parts[-1].strip()
                if answer:
                    # Log what we extracted (useful for debugging)
                    print(f"[EXTRACT] Found {close_tag}, extracted: {answer[:50]}...")
                    return answer
    
    # No recognized tags found
    # Fallback strategy: check if reply looks like it has reasoning
    lines = reply.strip().split('\n')
    
    # If multi-line and last line is short (likely an answer), use that
    if len(lines) > 3 and len(lines[-1]) < 50:
        # Looks like reasoning followed by short answer
        answer = lines[-1].strip()
        print(f"[EXTRACT] No tags found, using last line: {answer[:50]}...")
        return answer
    
    # If reply is very long (>500 chars), might be reasoning without tags
    # Take last paragraph
    if len(reply) > 500:
        paragraphs = [p.strip() for p in reply.split('\n\n') if p.strip()]
        if paragraphs:
            answer = paragraphs[-1]
            print(f"[EXTRACT] Long reply, using last paragraph: {answer[:50]}...")
            return answer
    
    # No extraction pattern matched, return as-is
    print(f"[EXTRACT] No reasoning pattern found, using full reply")
    return reply

# --------------------------------------------------------------------------------------
# Single-turn generation (appended to rolling context)
# --------------------------------------------------------------------------------------
def generate_reply(
    tok, mdl, prompt_text: str,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stop_regex: Optional[re.Pattern] = None,
    generation_config = None,  # For Qwen3 thinking mode
) -> str:
    inputs = tok(prompt_text, return_tensors="pt")
    if mdl.device.type != "cpu":
        inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
    
    # Build generation kwargs
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": (temperature > 0.0),
        "temperature": temperature,
        "top_p": top_p,
        "pad_token_id": tok.eos_token_id,
        "eos_token_id": tok.eos_token_id,
    }
    
    # Add generation_config if provided (for Qwen3 thinking mode)
    if generation_config is not None:
        gen_kwargs["generation_config"] = generation_config
    
    out = mdl.generate(**inputs, **gen_kwargs)
    text = tok.decode(out[0], skip_special_tokens=True)
    # Extract only what model added
    if text.startswith(prompt_text):
        gen = text[len(prompt_text):]
    else:
        gen = text
    if stop_regex:
        m = stop_regex.search(gen)
        if m:
            gen = gen[:m.start()]
    return gen.strip()

# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------
def parse_lengths(s: str, scaffold: str) -> List[int]:
    if scaffold == "baseline":
        return [1]
    if not s:
        return [6, 11, 16, 21]
    vals = sorted(set(int(x) for x in s.split(",")))
    for v in vals:
        if v not in (6, 11, 16, 21):
            raise ValueError(f"Invalid L={v} for scaffold={scaffold} (allowed: 6,11,16,21)")
    return vals

def load_final_items(path: Path, id_include: Optional[List[str]], num: Optional[int]) -> List[dict]:
    items = load_jsonl(path)
    if id_include:
        keep = set(id_include)
        items = [r for r in items if str(r.get("id")) in keep]
    if num is not None and num > 0:
        random.shuffle(items)
        items = items[:num]
    return items

def main():
    ap = argparse.ArgumentParser(description="Rolling-context runner for BoolQ-length evals")
    ap.add_argument("--scaffold", required=True, choices=SCAFFOLDS,
                    help="baseline|meta|semantic|underspecified|misleading")
    ap.add_argument("--lengths", default="",
                    help="Comma-separated. baseline forced to 1; others in {6,11,16,21}")
    ap.add_argument("--in-final", required=True,
                    help="Path to data/boolq_final.jsonl")
    ap.add_argument("--out-root", required=True,
                    help="Output root, e.g., runs/qwen2.5_meta")
    ap.add_argument("--num", type=int, default=None,
                    help="Use only N items (after id filter)")
    ap.add_argument("--id-include", default="",
                    help="Comma list of ids to include")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip if final response file already exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="Write prompts/transcripts without generation")
    ap.add_argument("--response-type", default="tri", choices=["tri", "bi"],
                    help="Response format: tri (YES/NO/IDK) or bi (YES/NO only)")

    # Inference knobs
    ap.add_argument("--model", default="phi4-mini", help="HF id or alias (e.g., phi4-mini)")
    ap.add_argument("--device", default="cpu", help="cpu|cuda|mps")
    ap.add_argument("--dtype", default="float16", help="compatibility flag; not strictly used here")
    ap.add_argument("--max-new-tokens-tasks", type=int, default=48, help="Per-turn cap for turns 1..L-1 (use 200 for reasoning models)")
    ap.add_argument("--max-new-tokens-final", type=int, default=16, help="Cap for final turn (use 800 for reasoning models)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--log-interval", type=int, default=50, help="Print every N items")
    
    # Reasoning model support
    ap.add_argument("--enable-thinking", action="store_true",
                    help="Enable thinking mode for Qwen3 models")
    ap.add_argument("--thinking-budget", type=int, default=None,
                    help="Thinking budget for Qwen3 (defaults to max-new-tokens-final)")
    ap.add_argument("--extract-reasoning", action="store_true",
                    help="Extract answer from reasoning tags (auto-detects common formats)")
    ap.add_argument("--reasoning-close-tag", type=str, default=None,
                    help="Custom closing tag for reasoning (e.g., '</think>' or '</reasoning>')")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="Load model in 4-bit quantization (for larger models)")

    args = ap.parse_args()

    scaffold = args.scaffold
    lengths = parse_lengths(args.lengths, scaffold)
    in_final = Path(args.in_final)
    out_root = Path(args.out_root)

    if scaffold == "baseline":
        lengths = [1]

    if not in_final.exists():
        raise FileNotFoundError(f"--in-final missing: {in_final}")

    id_include = [x for x in args.id_include.split(",") if x] if args.id_include else None
    items = load_final_items(in_final, id_include, args.num)

    print(f"[SETUP] scaffold={scaffold} lengths={lengths} response_type={args.response_type} items={len(items)} out_root={out_root}")

    # HF init (only if not dry-run)
    tok = mdl = generation_config = None
    if not args.dry_run:
        torch, AutoTokenizer, AutoModelForCausalLM = _lazy_hf()
        model_id = MODEL_ALIASES.get(args.model, args.model)
        print(f"[HF] Loading model: {model_id} on device={args.device}")
        
        # Load tokenizer
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # Load model with optional 4-bit quantization
        if args.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                print("[HF] Loading in 4-bit quantization...")
                mdl = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            except ImportError:
                print("[WARN] bitsandbytes not installed, loading without quantization")
                mdl = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
                if args.device == "cuda":
                    mdl = mdl.to("cuda")
                elif args.device == "mps":
                    mdl = mdl.to("mps")
                else:
                    mdl = mdl.to("cpu")
        else:
            mdl = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
            if args.device == "mps":
                mdl = mdl.to("mps")
            elif args.device == "cuda":
                mdl = mdl.to("cuda")
            else:
                mdl = mdl.to("cpu")
        
        # Setup Qwen3 thinking mode if requested
        if args.enable_thinking:
            try:
                from transformers import GenerationConfig
                thinking_budget = args.thinking_budget if args.thinking_budget else args.max_new_tokens_final
                generation_config = GenerationConfig(
                    enable_thinking=True,
                    thinking_budget=thinking_budget
                )
                print(f"[HF] Qwen3 thinking mode enabled (budget={thinking_budget})")
            except Exception as e:
                print(f"[WARN] Could not enable thinking mode: {e}")
                generation_config = None
        
        print("[HF] Ready.")

    # Stop regex: cut off if the model starts a new header we use
    stop_re = re.compile(r"\n+###\s*(User|Assistant)\b", re.IGNORECASE)

    manifest_rows: List[dict] = []
    N = len(items)

    for i, item in enumerate(items, 1):
        boolq_id = str(item.get("id"))
        gold_bool = bool(item.get("answer"))
        question = (item.get("corrected") or item.get("question") or "").strip()
        topic_primary = item.get("topic_primary")
        domain = item.get("domain")

        if (i == 1) or (i % args.log_interval == 0):
            print(f"[ITEM] {i}/{N} id={boolq_id}")

        for L in lengths:
            # Resolve template file(s)
            tmpl_path = resolve_template_path(scaffold, L, gold_bool, args.response_type)
            print(f"[TPL] L={L} -> {tmpl_path.name}")
            tmpl_rows = load_jsonl(tmpl_path)

            # Build placeholder map for this item
            subst = build_placeholder_map(item, scaffold)

            # Output layout - include response_type in filenames to distinguish runs
            run_dir = out_root / f"L{L}" / scaffold
            run_dir.mkdir(parents=True, exist_ok=True)
            base = f"{boolq_id}_L{L}_{scaffold}_{args.response_type}"
            out_prompt = run_dir / f"{base}.prompt.txt"
            out_resp   = run_dir / f"{base}.response.txt"
            out_json   = run_dir / f"{base}.json"

            if args.skip_existing and out_resp.exists():
                print(f"[SKIP] {base} (response exists)")
                continue

            # Rolling context transcript (string) + structured trace
            rolling = ""
            turns_trace: List[dict] = []
            t_start_all = time.time()

            # Generate per turn
            for tr in sorted(tmpl_rows, key=lambda r: int(r.get("turn", 0))):
                t_num = int(tr.get("turn", 0))
                origin = f"{tmpl_path.name}:turn{t_num}"
                prompt_raw = tr.get("prompt", "")
                prompt_text = strict_format(prompt_raw, subst, origin)

                # Build rolling context prompt with our simple headers
                user_block = f"### User\n{prompt_text}\n"
                full_prompt = f"{rolling}{user_block}### Assistant\n"

                # Which cap to use (final vs intermediate)
                is_final_turn = (t_num == L)
                cap = args.max_new_tokens_final if is_final_turn else args.max_new_tokens_tasks

                print(f"[TURN] id={boolq_id} L={L} t={t_num}/{L} (cap={cap})")
                t0 = time.time()

                if args.dry_run:
                    reply_full = "(dry-run)"
                    reply_cleaned = "(dry-run)"
                    gen_ms = 0.0
                else:
                    reply_full = generate_reply(
                        tok, mdl, full_prompt,
                        max_new_tokens=cap,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        stop_regex=stop_re,
                        generation_config=generation_config,
                    )
                    gen_ms = (time.time() - t0) * 1000.0
                    
                    # Extract answer from reasoning if requested
                    if args.extract_reasoning:
                        reply_cleaned = extract_answer_from_reasoning(reply_full, args.model)
                    else:
                        reply_cleaned = reply_full

                # Append CLEANED reply to rolling transcript (don't accumulate reasoning in context)
                rolling += f"{user_block}### Assistant\n{reply_cleaned}\n"

                # Save turn trace with both full and cleaned replies
                turns_trace.append({
                    "turn": t_num,
                    "prompt": prompt_text,
                    "reply": reply_full,  # Full reply with reasoning
                    "reply_cleaned": reply_cleaned,  # Answer only
                    "elapsed_ms": round(gen_ms, 1),
                })

            # Persist prompt (all user turns concatenated) and final response (last assistant line)
            prompt_only = "\n".join([f"[T{t['turn']}] {t['prompt']}" for t in turns_trace])
            save_text(out_prompt, prompt_only)

            # Use cleaned reply for final response (answer only, no reasoning)
            final_reply_cleaned = turns_trace[-1].get("reply_cleaned", turns_trace[-1]["reply"]) if turns_trace else ""
            save_text(out_resp, final_reply_cleaned)

            meta = {
                "boolq_id": boolq_id,
                "L": L,
                "scaffold": scaffold,
                "response_type": args.response_type,
                "template_file": str(tmpl_path),
                "mislead_branch": ("true" if (scaffold == "misleading" and gold_bool) else ("false" if scaffold == "misleading" else "")),
                "gold": "YES" if gold_bool else "NO",
                "question": question,
                "topic_primary": topic_primary,
                "domain": domain,
                "elapsed_s": round(time.time() - t_start_all, 3),
                "out_files": {
                    "prompt": str(out_prompt),
                    "response": str(out_resp),
                    "json": str(out_json),
                },
            }

            save_json(out_json, {"meta": meta, "turns": turns_trace, "transcript": rolling})
            print(f"[DONE] {base} in {meta['elapsed_s']}s; final len={len(final_reply_cleaned)}")
            manifest_rows.append(meta)

    # Manifest
    manifest_path = out_root / "_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[MANIFEST] {len(manifest_rows)} rows -> {manifest_path}")

if __name__ == "__main__":
    main()