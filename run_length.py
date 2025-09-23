#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_length.py — Length vs Veracity (BoolQ) runner, rolling context only.

Assets (per length L; file must contain exactly L lines):
  data/length/baseline/L1.jsonl
  data/length/meta/L{L}_info-meta.jsonl            # or L{L}_{scaffold}.jsonl or L{L}.jsonl
  data/length/semantic/L{L}_info-semantic.jsonl    # or L{L}_{scaffold}.jsonl or L{L}.jsonl
  data/length/underspecified/L{L}_info-underspecified.jsonl  # or L{L}_{scaffold}.jsonl or L{L}.jsonl

Usage examples:
  # Baseline (L=1 only)
  python run_length.py \
    --scaffold baseline \
    --lengths 1 \
    --in_enriched data/boolq_enriched.jsonl \
    --out_root runs/baseline_phi3 \
    --model microsoft/Phi-3-mini-4k-instruct \
    --device mps --dtype float32 \
    --max_new_tokens_tasks 64 --max_new_tokens_final 96 \
    --temperature 0 --stop_seq "### User"

  # Meta, any subset of {6,11,16,21}
  python run_length.py \
    --scaffold meta \
    --lengths 6,11,16,21 \
    --in_enriched data/boolq_enriched.jsonl \
    --out_root runs/meta_phi3 \
    --num 50 --skip_existing \
    --model microsoft/Phi-3-mini-4k-instruct \
    --device mps --dtype float32

  # Semantic
  python run_length.py \
    --scaffold semantic \
    --lengths 6,11,16,21 \
    --in_enriched data/boolq_enriched.jsonl \
    --out_root runs/semantic_phi3

  # Underspecified
  python run_length.py \
    --scaffold underspecified \
    --lengths 6,11,16,21 \
    --in_enriched data/boolq_enriched.jsonl \
    --out_root runs/underspecified_phi3

"""

import argparse, json, re, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -----------------------
# Small IO helpers
# -----------------------
def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows

def read_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sanitize_id(s: Optional[str], fallback: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9_-]+", "_", s).strip("_")
    return s if s else fallback

# -----------------------
# ID include parsing (e.g., 'dev_0063-dev_0100,dev_0123')
# -----------------------
def expand_id_spec(spec: Optional[str]) -> set:
    out = set()
    if not spec:
        return out
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            a, b = part.split("-", 1)
            m1 = re.match(r"^(.*?)(\d+)$", a)
            m2 = re.match(r"^(.*?)(\d+)$", b)
            if m1 and m2 and m1.group(1) == m2.group(1):
                prefix = m1.group(1)
                start, end = int(m1.group(2)), int(m2.group(2))
                width = max(len(m1.group(2)), len(m2.group(2)))
                for n in range(min(start, end), max(start, end) + 1):
                    out.add(f"{prefix}{n:0{width}d}")
            else:
                out.add(a); out.add(b)
        else:
            out.add(part)
    return out

# -----------------------
# Asset loader (one JSONL per length)
# -----------------------
def load_length_file(base_dir: Path, scaffold: str, L: int) -> List[dict]:
    if scaffold not in ("baseline", "meta", "semantic", "underspecified"):
        raise ValueError("scaffold must be 'baseline', 'meta', 'semantic', or 'underspecified'")

    # Enforce scaffold/length rules here too (defensive)
    if scaffold == "baseline" and L != 1:
        raise ValueError("Baseline scaffold only supports L=1.")
    if scaffold in ("meta", "semantic", "underspecified") and L == 1:
        raise ValueError(f"{scaffold} scaffold does not include L=1 (run baseline for L=1).")

    # Candidate filenames in priority order
    if scaffold == "baseline":
        candidates = [
            base_dir / "baseline" / "L1_info-baseline.jsonl",
            base_dir / "baseline" / "L1_baseline.jsonl",
            base_dir / "baseline" / "L1.jsonl",
        ]
    else:
        candidates = [
            base_dir / scaffold / f"L{L}_info-{scaffold}.jsonl",
            base_dir / scaffold / f"L{L}_{scaffold}.jsonl",
            base_dir / scaffold / f"L{L}.jsonl",
        ]

    for c in candidates:
        if c.exists():
            rows = read_jsonl(c)
            if len(rows) != L:
                raise ValueError(f"{c} expected {L} lines (turns) but found {len(rows)}")
            rows.sort(key=lambda r: r.get("turn", 0))
            return rows

    # Fallback: any file starting with L{L}_ under the scaffold folder
    for p in (base_dir / scaffold).glob(f"L{L}_*.jsonl"):
        rows = read_jsonl(p)
        rows.sort(key=lambda r: r.get("turn", 0))
        return rows

    raise FileNotFoundError(f"No asset file found for scaffold={scaffold} length={L} under {base_dir/scaffold}")

# -----------------------
# Placeholder substitution for SEMANTIC prompts
# -----------------------
def build_placeholder_map(enriched: dict) -> Dict[str, str]:
    topic = enriched.get("topic_primary") or enriched.get("topic") or ""
    related = enriched.get("topic_related") or []
    rel = [str(x) for x in related if isinstance(x, (str, int, float))]

    def get_rel(idx: int, fallback: str = "") -> str:
        if idx < len(rel):
            return rel[idx]
        return fallback or (rel[-1] if rel else topic or "")

    return {
        "{QUESTION}": enriched.get("corrected") or enriched.get("question") or "",
        "{topic_primary}": topic,
        "{related_a}": get_rel(0),
        "{related_b}": get_rel(1),
        "{related_c}": get_rel(2),
        "{related_d}": get_rel(3),
    }

def substitute_placeholders(template: str, mapping: Dict[str, str]) -> str:
    out = template
    for k, v in mapping.items():
        out = out.replace(k, v)
    return out

# -----------------------
# Simple HF causal runner (same as before)
# -----------------------
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelRunner:
    def __init__(self, model_key_or_id: str, device: str = "auto", dtype: str = "auto"):
        cfg = MODEL_REGISTRY.get(model_key_or_id, {"backend": "hf_causal", "model_id": model_key_or_id})
        if cfg.get("backend") != "hf_causal":
            raise NotImplementedError("Only HF causal backend is implemented here.")
        self.model_id = cfg["model_id"]
        self.device = device
        self.dtype = dtype
        self._init_hf()

    def _init_hf(self):
        torch_dtype = {"auto": None, "float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[self.dtype]
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch_dtype or "auto")
        if self.device == "auto":
            self._input_device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        elif self.device == "cuda":
            self._input_device = torch.device("cuda")
        elif self.device == "mps":
            self._input_device = torch.device("mps")
        else:
            self._input_device = torch.device("cpu")
        self.model.to(self._input_device)

    def generate(self, prompt: str, max_new_tokens: int, temperature: float = 0.0, stop: Optional[List[str]] = None) -> str:
        with torch.inference_mode():
            enc = self.tok(prompt, return_tensors="pt")
            inputs = enc.to(self._input_device) if self.device != "auto" else {k: v.to(self._input_device) for k, v in enc.items()}
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False if (temperature is None or temperature == 0) else True,
                temperature=temperature if (temperature and temperature > 0) else 1.0,
                pad_token_id=self.tok.pad_token_id,
            )[0]
        in_len = inputs["input_ids"].shape[-1]
        gen_only_ids = output_ids[in_len:]
        reply = self.tok.decode(gen_only_ids, skip_special_tokens=True).strip()
        if stop:
            for token in stop:
                i = reply.find(token)
                if i != -1:
                    reply = reply[:i].strip()
                    break
        return reply

MODEL_REGISTRY = {
    "phi3mini": {"backend": "hf_causal", "model_id": "microsoft/Phi-3-mini-4k-instruct"},
    # Add others as needed
}

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Length vs Veracity runner (rolling context).")
    ap.add_argument("--scaffold", choices=["baseline","meta","semantic","underspecified"], required=True)
    ap.add_argument("--lengths", required=True, help="Comma list of lengths, e.g., '1' (baseline) or '6,11,16,21'")
    ap.add_argument("--in_enriched", required=True, help="Path to boolq_enriched.jsonl (authoritative source incl. gold 'answer').")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--num", type=int, default=None, help="Cap number of items after filtering")
    ap.add_argument("--id_include", default=None, help="Comma list/ranges of BoolQ ids, e.g. 'dev_0001-dev_0100'")
    ap.add_argument("--skip_existing", action="store_true")
    # model/decoding
    ap.add_argument("--model", default="phi3mini", help="Key or HF model id")
    ap.add_argument("--device", choices=["cpu","cuda","mps","auto"], default="mps")
    ap.add_argument("--dtype", choices=["auto","float32","float16","bfloat16"], default="float16")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens_tasks", type=int, default=64)
    ap.add_argument("--max_new_tokens_final", type=int, default=96)
    ap.add_argument("--stop_seq", default="### User")
    # assets root (allow override)
    ap.add_argument("--assets_root", default="data/length", help="Base folder for length assets.")
    args = ap.parse_args()

    # Parse lengths
    try:
        lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    except Exception:
        raise SystemExit("--lengths must be comma-separated integers, e.g., '1' or '6,11,16,21'")

    # Enforce scaffold/length policy
    if args.scaffold == "baseline":
        if set(lengths) != {1}:
            raise SystemExit("For scaffold=baseline you must set --lengths 1 (and only 1).")
    else:
        if 1 in lengths:
            raise SystemExit(f"For scaffold={args.scaffold}, remove L=1 (run baseline separately).")

    # Load enriched only
    enr_path = Path(args.in_enriched)
    if not enr_path.exists():
        raise SystemExit(f"Not found: {enr_path}")
    enriched = read_jsonl(enr_path)
    if not enriched:
        raise SystemExit("No rows found in --in_enriched.")

    # Filter by id if requested
    items = enriched
    if args.id_include:
        wanted = expand_id_spec(args.id_include)
        items = [e for e in items if (e.get("id") in wanted)]
        print(f"[FILTER] id_include -> {len(items)} items")

    # Cap by --num
    if args.num is not None and args.num > 0:
        items = items[: args.num]
        
    logging.info("Loaded {} enriched items, leaving {} items after filtering".format(len(enriched), len(items)))

    # Load model
    logging.info("Loading model {} on {} ({}) .... ".format(args.model, args.device, args.dtype))
    runner = ModelRunner(args.model, device=args.device, dtype=args.dtype)
    # Warmup
    _ = runner.generate("### User\nok\n### Assistant\n", max_new_tokens=1, temperature=0.0, stop=None)

    # Stop list (trim reply only)
    stop_list = None
    if args.stop_seq and args.stop_seq.strip():
        tok = args.stop_seq.strip()
        stop_list = ["\n" + tok, tok]

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    assets_root = Path(args.assets_root)

    # manifest out path
    mpth = out_root / "_manifest.json"
    if mpth.exists():
        logging.info("Loading manifest from {}...".format(mpth.as_posix()))
        manifest = read_json(mpth)
    else:
        manifest = {
        "mode": "length",
        # "scaffold": args.scaffold,
        # "lengths": lengths,
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "decoding": {"temperature": args.temperature,
                     "max_new_tokens_tasks": args.max_new_tokens_tasks,
                     "max_new_tokens_final": args.max_new_tokens_final,
                     "stop_seq": args.stop_seq},
        "assets_root": str(assets_root.resolve()),
        "items": {},
    }

    logging.info("**** lengths = {},  scaffold = {} ****".format(lengths, args.scaffold))

    for L in lengths:
        fout = out_root.joinpath("L{}_{}.json".format(L, args.scaffold))
        logging.info("Out path: {}".format(fout.as_posix()))

        turns = load_length_file(assets_root, args.scaffold, L)
        tmp_out_items = []
        for ix, erow in tqdm(enumerate(items), total=len(items), desc="L{}".format(L)):

            boolq_id_raw = erow.get("id")
            boolq_id = sanitize_id(boolq_id_raw, "unknown")
            q_raw = erow.get("question", "")
            q_corr = erow.get("corrected") or q_raw

            # check if processed
            if fout.name in manifest["items"].get(boolq_id, []):
                # pbar.update(1)
                continue

            enriched_meta = {
                "id": boolq_id_raw,
                "question": q_raw,
                "corrected": q_corr,
                "topic_primary": erow.get("topic_primary"),
                "topic_related": erow.get("topic_related") or [],
            }
            ph_map = build_placeholder_map(enriched_meta)

            # Build rolling conversation
            transcript = []
            rolling = ""
            t_start = time.time()
            for t in range(1, L + 1):
                entry = turns[t-1]
                prompt_tpl = entry.get("prompt", "")

                # Substitute placeholders
                if args.scaffold == "semantic":
                    prompt_text = substitute_placeholders(prompt_tpl, ph_map)
                else:
                    # baseline/meta/underspecified: only {QUESTION} replacement (harmless for turns without it)
                    prompt_text = substitute_placeholders(prompt_tpl, {"{QUESTION}": q_corr})

                # Construct prompt with rolling history
                full_prompt = f"{rolling}### User\n{prompt_text}\n### Assistant\n"
                is_final = (t == L)
                cap = args.max_new_tokens_final if is_final else args.max_new_tokens_tasks

                reply = runner.generate(full_prompt, max_new_tokens=cap,
                                        temperature=args.temperature, stop=stop_list)

                transcript.append({
                    "turn": t, 
                    "user": prompt_text, 
                    "model": reply
                })

                rolling += f"### User\n{prompt_text}\n### Assistant\n{reply}\n"

                _token_count = full_prompt.count(" ") + reply.count(" ") + 2
                if _token_count >= 4000:
                    logging.warning("Excessive tokens: {}, {}/{}, boolq_id = {}".format(_token_count, t, L, boolq_id))

            elapsed = time.time() - t_start

            obj = {
                "boolq_id": boolq_id,
                "question": q_corr,
                "gold_answer": erow.get("answer"),
                "enriched": {"topic_primary": erow.get("topic_primary"),
                             "topic_related": erow.get("topic_related") or []},
                "timing_s": round(elapsed, 3),
                "transcript": transcript,
            }
            tmp_out_items.append(obj)
            # 更新manifest
            manifest["items"][boolq_id] = manifest["items"].get(boolq_id, []) + [fout.name]
            # cache 定期写出
            if (ix + 1) % 600: #! debug
                if fout.exists():  # 直接写出
                    with open(fout, "a") as f:
                        f.writelines(json.dumps(tmp_item) + "\n" for tmp_item in tmp_out_items)

                else:  # 写出headline
                    common = [{
                    "mode": "length",
                    "scaffold": args.scaffold,
                    "length": L,
                    "model": {"id": args.model, "device": args.device, "dtype": args.dtype},
                    "decoding": {"temperature": args.temperature,
                                "max_new_tokens_tasks": args.max_new_tokens_tasks,
                                "max_new_tokens_final": args.max_new_tokens_final,
                                "stop_seq": args.stop_seq}
                    }]
                    with open(fout, "w") as f:
                        f.write(json.dumps(common) +"\n")
                write_json(mpth, manifest) 
                tmp_out_items.clear()

        if tmp_out_items:
            if fout.exists():  # 直接写出
                with open(fout, "a") as f:
                    f.writelines(json.dumps(tmp_item) + "\n" for tmp_item in tmp_out_items)

            else:  # 写出headline
                common = [{
                "mode": "length",
                "scaffold": args.scaffold,
                "length": L,
                "model": {"id": args.model, "device": args.device, "dtype": args.dtype},
                "decoding": {"temperature": args.temperature,
                            "max_new_tokens_tasks": args.max_new_tokens_tasks,
                            "max_new_tokens_final": args.max_new_tokens_final,
                            "stop_seq": args.stop_seq}
                }]
                with open(fout, "w") as f:
                    f.write(json.dumps(common) +"\n")
            write_json(mpth, manifest)
            tmp_out_items.clear()

    write_json(mpth, manifest)
if __name__ == "__main__":
    main()