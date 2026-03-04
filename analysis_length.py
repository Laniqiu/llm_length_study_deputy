#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_length.py

Analysis script that:
1. Reads conversation JSONs directly from runs/model/scaffold/ structure
2. Parses responses and creates detailed_rows.csv (for inspection)
3. Performs full analysis (regression, plots, flat baselines)

Processes 5 models: phi4, deepseek7b, qwen2.5, llama3.1, mistral7b
"""

from pathlib import Path
import json
import re
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

# ---------- Configuration ----------
PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs"
DATA_DIR = PROJECT_ROOT / "data"
FINAL_JSONL = DATA_DIR / "boolq_final.jsonl"

# Model configurations
MODEL_CONFIGS = {
    "phi4": {"folder": RUNS_DIR / "phi4", "display_name": "Phi-4", "color": "#1f77b4"},
    "deepseek7b": {"folder": RUNS_DIR / "deepseek7b", "display_name": "DeepSeek-7B", "color": "#ff7f0e"},
    "qwen2.5": {"folder": RUNS_DIR / "qwen2.5", "display_name": "Qwen2.5", "color": "#2ca02c"},
    "llama3.1": {"folder": RUNS_DIR / "llama3.1", "display_name": "Llama-3.1-8B", "color": "#d62728"},
    "mistral7b": {"folder": RUNS_DIR / "mistral7b", "display_name": "Mistral-7B", "color": "#9467bd"},
}

REGRESSION_DIR = RUNS_DIR / "regression"
REGRESSION_DIR.mkdir(parents=True, exist_ok=True)

COMBINED_DIR = RUNS_DIR / "combined_plots"
COMBINED_DIR.mkdir(parents=True, exist_ok=True)

VALID_SCAFFOLDS = ["meta", "semantic", "underspecified", "misleading"]
VALID_L = [6, 11, 16, 21]
VALID_L_WITH_BASELINE = [1, 6, 11, 16, 21]
ACC_ALL = ["Incorrect", "Correct", "IDK"]
ACC_ALL_WITH_NC = ["Incorrect", "Correct", "IDK", "NC"]  # For per-model plots that show non-compliant

MARKERS_CONDITIONS = {"meta": "o", "semantic": "s", "underspecified": "D", "misleading": "X"}
DISPLAY_NAMES_SCAFFOLDS = {"meta": "Meta", "semantic": "Semantic", "underspecified": "Underspecified", "misleading": "Misleading"}

# ---------- Response Parsing ----------
_WORD_RE = re.compile(r"[A-Za-z]+")
_IDK_PATTERNS = [r"\bi\s*(?:do\s*not|don'?t)\s*know\b", r"\bnot\s*sure\b", r"\bunsure\b", r"\bunknown\b", r"\bcannot\s*(?:tell|determine)\b", r"\bno\s*idea\b"]
# YES patterns: prefer start of text, but also match anywhere as fallback
_YES_PATTERNS = [r"^yes\b", r"^\by\b", r"\btrue\b", r"\bcorrect\b", r"\byes\b"]
# NO patterns: prefer start of text, but also match anywhere as fallback  
_NO_PATTERNS = [r"^no\b", r"^\bn\b", r"\bfalse\b", r"\bincorrect\b", r"\bno\b"]

def first_token_english_yes_no(text: str) -> tuple:
    s = (text or "").lstrip()
    m = _WORD_RE.search(s)
    if not m:
        return ("", False)
    tok = m.group(0).lower()
    return (tok, True) if tok in ("yes", "no") else (tok, False)

def extract_decision_lenient(text: str) -> tuple:
    s = (text or "").strip().lower()
    s = re.sub(r"^\s*(?:answer|final answer)\s*[:\-–]\s*", "", s)
    
    def _search_any(patterns):
        first = None
        for pat in patterns:
            m = re.search(pat, s, flags=re.I)
            if m and (first is None or m.start() < first.start()):
                first = m
        return first
    
    candidates = []
    for label, patterns in [("idk", _IDK_PATTERNS), ("yes", _YES_PATTERNS), ("no", _NO_PATTERNS)]:
        m = _search_any(patterns)
        if m:
            candidates.append((label, m))
    
    if not candidates:
        return ("other", "")
    
    label, match = min(candidates, key=lambda kv: kv[1].start())
    return (label, s[match.start():match.end()])

# ---------- Data Loading ----------
def load_final_ids_and_gold(path: Path) -> tuple:
    """Load BoolQ IDs and their true gold labels from final JSONL."""
    ids = set()
    gold_map = {}
    if not path.exists():
        return ids, gold_map
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if "id" in obj:
                    boolq_id = str(obj["id"])
                    ids.add(boolq_id)
                    # True gold label from boolq_final.jsonl
                    if "answer" in obj:
                        gold_map[boolq_id] = 1 if obj["answer"] == True else 0
    return ids, gold_map

def parse_conversation_file(conv_path: Path, scaffold: str, gold_map: dict) -> list:
    """Parse a conversation file that contains multiple conversations.
    
    Line 1: Metadata (applies to all conversations)
    Lines 2+: Individual conversations (one JSON per line)
    
    Uses true gold labels from gold_map, not from conversation JSON.
    """
    with conv_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        return []
    
    # Line 1: metadata (has 'length', 'scaffold')
    metadata = json.loads(lines[0])
    length = int(metadata.get("length", 0))
    
    # Lines 2+: individual conversations
    conversations = []
    for line in lines[1:]:
        if not line.strip():
            continue
        
        try:
            conv = json.loads(line)
            turns = conv.get("turns", [])
            if not turns:
                continue
            
            boolq_id = conv.get("boolq_id", "")
            
            # Use TRUE gold label from boolq_final.jsonl
            gold_numeric = gold_map.get(boolq_id)
            if gold_numeric is None:
                # If not in gold_map, fall back to JSON (but this shouldn't happen)
                gold_numeric = 1 if conv.get("gold", "").upper() == "YES" else 0
            
            # Convert to yes/no for readability
            gold = "yes" if gold_numeric == 1 else "no"
            
            # Extract reply and clean template leakage
            reply = turns[-1].get("reply", "")
            
            # Remove markdown bold markers (Llama often uses **YES** or **NO**)
            reply = reply.replace("**", "").strip()
            
            # Truncate at chat template markers
            markers = [
                "###Human:", "###Human", "### Human:", "### Human",
                "###User:", "###User", "### User:", "### User",
                "###Assistant:", "###Assistant", "### Assistant:", "### Assistant",
                "###",  # Standalone ### marker (must come after longer patterns)
                "Human:", "User:", "Assistant:"
            ]
            for marker in markers:
                if marker in reply:
                    reply = reply.split(marker)[0].strip()
                    break
            
            # Truncate at "assistant" (Llama concatenates this)
            # Handles: "YESassistant" and "YESassistant\nMore text..."
            if "assistant" in reply.lower():
                pos = reply.lower().find("assistant")
                reply = reply[:pos].strip()
            
            # Truncate at "human" (Qwen concatenates this)
            # Handles: "NOHuman", "YESHumanHumanHuman", "NOHuman: Can you..."
            if "human" in reply.lower():
                pos = reply.lower().find("human")
                reply = reply[:pos].strip()
            
            # Truncate at "user" (some models concatenate this)
            if "user" in reply.lower():
                pos = reply.lower().find("user")
                reply = reply[:pos].strip()
            
            # Strip common prefixes that obscure yes/no
            # "Ready. NO." -> "NO.", "The answer is: NO." -> "NO.", "Answer: YES" -> "YES"
            prefix_patterns = [
                r"^ready\.\s*",
                r"^the\s+answer\s+is\s*[:\-–]?\s*",
                r"^answer\s*[:\-–]?\s*",
                r"^my\s+answer\s+is\s*[:\-–]?\s*",
                r"^response\s*[:\-–]?\s*",
            ]
            for pattern in prefix_patterns:
                reply = re.sub(pattern, "", reply, flags=re.IGNORECASE).strip()
            
            # Get final turn timing (convert ms to seconds)
            # This measures time to answer the BoolQ question specifically, not cumulative time
            final_turn_elapsed_ms = turns[-1].get("elapsed_ms", 0)
            timing_s = final_turn_elapsed_ms / 1000.0 if final_turn_elapsed_ms else 0.0
            
            row_dict = {
                "scaffold": scaffold,
                "L": length,
                "boolq_id": boolq_id,
                "gold": gold,  # yes/no instead of 0/1
                "reply": reply,  # Cleaned reply
                "timing_s": timing_s,  # Final turn timing only
            }
            conversations.append(row_dict)
        except json.JSONDecodeError:
            continue
    
    return conversations

def score_response(row_dict: dict) -> dict:
    """Score a response and determine acc_type (Correct/Incorrect/IDK/NC).
    
    Simplified output with only essential columns.
    """
    reply = row_dict["reply"]
    gold = row_dict["gold"]  # Now 'yes' or 'no' string
    
    # Extract first token
    first_tok, compliant = first_token_english_yes_no(reply)
    
    # Lenient parsing
    pred_label, pred_raw = extract_decision_lenient(reply)
    
    # Determine acc_type
    if pred_label == "idk":
        acc_type = "IDK"
    elif pred_label in ("yes", "no"):
        # Check if prediction matches gold
        if pred_label == gold:
            acc_type = "Correct"
        else:
            acc_type = "Incorrect"
    else:
        # pred_label is "other" - non-compliant
        acc_type = "NC"
    
    # Update row with simplified columns only
    row_dict.update({
        "first_token": first_tok,
        "pred_label": pred_label,
        "acc_type": acc_type,
    })
    
    return row_dict

def load_scaffold_data(model_folder: Path, scaffold: str, keep_ids: set, gold_map: dict) -> list:
    scaffold_dir = model_folder / scaffold
    manifest_path = scaffold_dir / "_manifest.jsonl"
    
    if not manifest_path.exists():
        print(f"  [WARN] No manifest: {manifest_path}")
        return []
    
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    items = manifest.get("items", {})
    print(f"  Loading {scaffold}: {len(items)} boolq_ids in manifest")
    
    # Get unique conversation files - each boolq_id maps to MULTIPLE files (one per length)
    unique_files = set()
    for conv_files in items.values():
        if conv_files:
            # Add ALL files, not just the first one
            for filename in conv_files:
                unique_files.add(filename)
    
    print(f"  Found {len(unique_files)} unique conversation files")
    
    # Load all conversations from all files
    rows = []
    for filename in unique_files:
        conv_path = scaffold_dir / filename
        if conv_path.exists():
            # Each file contains multiple conversations
            conversations = parse_conversation_file(conv_path, scaffold, gold_map)
            for row_dict in conversations:
                # Filter by keep_ids
                if not keep_ids or row_dict["boolq_id"] in keep_ids:
                    rows.append(score_response(row_dict))
        else:
            print(f"  [WARN] File not found: {filename}")
    
    print(f"  Loaded {scaffold}: {len(rows)} conversations (after filtering)")
    return rows

def build_master_dataframe(model_name: str, model_folder: Path, keep_ids: set, gold_map: dict) -> pd.DataFrame:
    print(f"\n--- Processing {model_name} ---")
    
    baseline_rows = load_scaffold_data(model_folder, "baseline", keep_ids, gold_map)
    if not baseline_rows:
        print(f"  [WARN] No baseline data")
        return pd.DataFrame()
    
    baseline_df = pd.DataFrame(baseline_rows)
    
    # Debug: show what L values we have
    if "L" in baseline_df.columns:
        l_values = baseline_df["L"].unique()
        print(f"  Baseline L values found: {sorted(l_values)}")
    
    baseline_l1 = baseline_df[baseline_df["L"] == 1].copy()
    
    if baseline_l1.empty:
        print(f"  [WARN] No L=1 baseline (total baseline rows: {len(baseline_df)})")
        return pd.DataFrame()
    
    print(f"  Found {len(baseline_l1)} baseline L=1 observations")
    
    # Write baseline CSV with columns in preferred order
    (model_folder / "baseline").mkdir(exist_ok=True)
    baseline_df = baseline_df[["scaffold", "L", "boolq_id", "gold", "reply", "first_token", "pred_label", "acc_type", "timing_s"]]
    baseline_df.to_csv(model_folder / "baseline" / "detailed_rows.csv", index=False)
    print(f"  [WRITE] baseline/detailed_rows.csv")
    
    all_parts = []
    for scaffold in VALID_SCAFFOLDS:
        scaffold_baseline = baseline_l1.copy()
        scaffold_baseline["scaffold"] = scaffold
        all_parts.append(scaffold_baseline)
        
        scaffold_rows = load_scaffold_data(model_folder, scaffold, keep_ids, gold_map)
        if scaffold_rows:
            scaffold_df = pd.DataFrame(scaffold_rows)
            (model_folder / scaffold).mkdir(exist_ok=True)
            # Reorder columns
            scaffold_df = scaffold_df[["scaffold", "L", "boolq_id", "gold", "reply", "first_token", "pred_label", "acc_type", "timing_s"]]
            scaffold_df.to_csv(model_folder / scaffold / "detailed_rows.csv", index=False)
            print(f"  [WRITE] {scaffold}/detailed_rows.csv")
            all_parts.append(scaffold_df)
    
    merged = pd.concat(all_parts, ignore_index=True)
    
    # Keep all acc_type values including NC (for per-model plots)
    # Regression and old plots will filter to ACC_ALL as needed
    merged = merged[merged["acc_type"].isin(ACC_ALL_WITH_NC)].copy()
    
    print(f"  Scaffolds: {sorted(merged['scaffold'].unique())}")
    print(f"  Lengths: {sorted(merged['L'].unique())}")
    
    return merged

# ---------- Analysis Functions ----------
def percent_table(grouped: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    totals = grouped.groupby(group_cols)["n"].transform("sum")
    grouped["percent"] = grouped["n"] / totals
    return grouped

def calculate_flat_dummy_for_viz(master_df: pd.DataFrame) -> pd.DataFrame:
    l1_data = master_df[master_df["L"] == 1].copy()
    if l1_data.empty:
        return pd.DataFrame()
    
    total = len(l1_data)
    results = []
    for L in VALID_L_WITH_BASELINE:
        for acc_type in ACC_ALL:
            count = len(l1_data[l1_data["acc_type"] == acc_type])
            results.append({"L": L, "acc_type": acc_type, "percent": count / total if total > 0 else 0})
    return pd.DataFrame(results)

def plot_combined_by_response_type(master_dfs: dict, response_type: str, flat_dummies: dict):
    fig, ax = plt.subplots(figsize=(7.2, 6))
    
    for model_name, merged in master_dfs.items():
        if merged.empty:
            continue
        
        len_df = merged[merged["scaffold"].isin(VALID_SCAFFOLDS)].copy()
        len_df["L"] = len_df["L"].astype(int)
        
        grouped = len_df.groupby(["scaffold", "L", "acc_type"]).size().reset_index(name="n")
        pct = percent_table(grouped, ["scaffold", "L"])
        response_data = pct[pct["acc_type"] == response_type]
        
        model_color = MODEL_CONFIGS[model_name]["color"]
        
        if flat_dummies and model_name in flat_dummies:
            flat_response = flat_dummies[model_name][flat_dummies[model_name]["acc_type"] == response_type]
            if not flat_response.empty:
                ax.plot(flat_response["L"], flat_response["percent"], linestyle='--', color=model_color, linewidth=2.5, alpha=0.7)
        
        for scaffold in VALID_SCAFFOLDS:
            sub = response_data[response_data["scaffold"] == scaffold]
            if not sub.empty:
                ax.plot(sub["L"], sub["percent"], marker=MARKERS_CONDITIONS[scaffold], color=model_color, linewidth=2.0, markersize=7, alpha=0.8)
    
    ax.set_xlabel("Length (L)", fontsize=13)
    ax.set_ylabel(f"Percent {response_type}", fontsize=13)
    ax.set_xticks(VALID_L_WITH_BASELINE)
    ax.set_ylim(0, 0.8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.set_title(f"{response_type} Responses by Length", fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    model_handles = [Line2D([0], [0], color=MODEL_CONFIGS[m]["color"], linewidth=2.5, label=MODEL_CONFIGS[m]["display_name"]) for m in MODEL_CONFIGS.keys()]
    baseline_handle = Line2D([0], [0], linestyle='--', color='black', linewidth=2.5, label='Flat Baseline (L=1)')
    condition_handles = [Line2D([0], [0], marker=MARKERS_CONDITIONS[c], color='black', linewidth=0, markersize=8, label=DISPLAY_NAMES_SCAFFOLDS[c]) for c in VALID_SCAFFOLDS]
    
    first_legend = ax.legend(handles=model_handles, loc='upper left', title="Models", frameon=True, fontsize=11)
    ax.add_artist(first_legend)
    ax.legend(handles=[baseline_handle] + condition_handles, loc='upper right', title="Conditions", frameon=True, fontsize=11)
    
    fig.tight_layout()
    fig.savefig(COMBINED_DIR / f"combined_{response_type.lower()}_by_condition.png", dpi=200)
    plt.close(fig)
    print(f"[COMBINED PLOT] Wrote {response_type}")

def plot_per_model_all_outcomes(model_name: str, merged: pd.DataFrame):
    """Create one plot per model showing all 4 outcomes (including NC) with scaffold markers."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Outcome colors
    outcome_colors = {
        "Correct": "#2ca02c",    # green
        "Incorrect": "#d62728",  # red
        "IDK": "#ff7f0e",        # orange
        "NC": "#7f7f7f"          # gray
    }
    
    model_display = MODEL_CONFIGS[model_name]["display_name"]
    
    # Calculate flat baseline for this model (including NC)
    l1_data = merged[merged["L"] == 1].copy()
    flat_baseline = {}
    if not l1_data.empty:
        total = len(l1_data)
        for acc_type in ACC_ALL_WITH_NC:
            count = len(l1_data[l1_data["acc_type"] == acc_type])
            flat_baseline[acc_type] = count / total if total > 0 else 0
    
    # Plot each outcome type
    for outcome in ACC_ALL_WITH_NC:
        outcome_color = outcome_colors[outcome]
        
        # Plot flat baseline for this outcome (dashed)
        if outcome in flat_baseline:
            baseline_percent = flat_baseline[outcome]
            ax.plot(VALID_L_WITH_BASELINE, [baseline_percent] * len(VALID_L_WITH_BASELINE),
                   linestyle='--', color=outcome_color, linewidth=2.5, alpha=0.7)
        
        # Plot each scaffold for this outcome (solid with markers)
        len_df = merged[merged["scaffold"].isin(VALID_SCAFFOLDS)].copy()
        len_df["L"] = len_df["L"].astype(int)
        
        grouped = len_df.groupby(["scaffold", "L", "acc_type"]).size().reset_index(name="n")
        pct = percent_table(grouped, ["scaffold", "L"])
        outcome_data = pct[pct["acc_type"] == outcome]
        
        for scaffold in VALID_SCAFFOLDS:
            sub = outcome_data[outcome_data["scaffold"] == scaffold]
            if not sub.empty:
                ax.plot(sub["L"], sub["percent"], 
                       marker=MARKERS_CONDITIONS[scaffold], color=outcome_color,
                       linewidth=2.0, markersize=7, alpha=0.8)
    
    ax.set_xlabel("Length (L)", fontsize=13)
    ax.set_ylabel("Percent", fontsize=13)
    ax.set_xticks(VALID_L_WITH_BASELINE)
    ax.set_ylim(0, 0.8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.set_title(f"{model_display}: All Outcomes by Scaffold", fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Create legend
    # Outcome colors
    outcome_handles = [Line2D([0], [0], color=outcome_colors[out], linewidth=2.5, 
                             label=out) for out in ACC_ALL_WITH_NC]
    # Scaffold markers
    scaffold_handles = [Line2D([0], [0], marker=MARKERS_CONDITIONS[s], color='black', 
                               linewidth=0, markersize=8, label=DISPLAY_NAMES_SCAFFOLDS[s]) 
                       for s in VALID_SCAFFOLDS]
    # Baseline
    baseline_handle = Line2D([0], [0], linestyle='--', color='black', linewidth=2.5, 
                            label='Flat Baseline')
    
    # Two legends
    first_legend = ax.legend(handles=outcome_handles, loc='upper left', 
                            title="Outcomes", frameon=True, fontsize=10)
    ax.add_artist(first_legend)
    ax.legend(handles=[baseline_handle] + scaffold_handles, loc='upper right', 
             title="Scaffolds", frameon=True, fontsize=10)
    
    fig.tight_layout()
    output_path = COMBINED_DIR / f"{model_name}_all_outcomes.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[PER-MODEL PLOT] Wrote {model_name}_all_outcomes.png")

def plot_stacked_combined(master_dfs: dict, flat_dummies: dict):
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 12), sharex=True)
    response_types = ["Incorrect", "Correct", "IDK"]
    panel_labels = ["a)", "b)", "c)"]
    
    for idx, (response_type, panel_label) in enumerate(zip(response_types, panel_labels)):
        ax = axes[idx]
        
        for model_name, merged in master_dfs.items():
            if merged.empty:
                continue
            
            len_df = merged[merged["scaffold"].isin(VALID_SCAFFOLDS)].copy()
            len_df["L"] = len_df["L"].astype(int)
            grouped = len_df.groupby(["scaffold", "L", "acc_type"]).size().reset_index(name="n")
            pct = percent_table(grouped, ["scaffold", "L"])
            response_data = pct[pct["acc_type"] == response_type]
            
            model_color = MODEL_CONFIGS[model_name]["color"]
            
            if flat_dummies and model_name in flat_dummies:
                flat_response = flat_dummies[model_name][flat_dummies[model_name]["acc_type"] == response_type]
                if not flat_response.empty:
                    ax.plot(flat_response["L"], flat_response["percent"], linestyle='--', color=model_color, linewidth=2.5, alpha=0.7)
            
            for scaffold in VALID_SCAFFOLDS:
                sub = response_data[response_data["scaffold"] == scaffold]
                if not sub.empty:
                    ax.plot(sub["L"], sub["percent"], marker=MARKERS_CONDITIONS[scaffold], color=model_color, linewidth=2.0, markersize=7, alpha=0.8)
        
        ax.text(-0.12, 1.0, panel_label, transform=ax.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
        ax.set_ylabel(f"Percent {response_type}", fontsize=13)
        ax.set_ylim(0, 0.8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        ax.set_title(f"{response_type} Responses by Length", fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            model_handles = [Line2D([0], [0], color=MODEL_CONFIGS[m]["color"], linewidth=2.5, label=MODEL_CONFIGS[m]["display_name"]) for m in MODEL_CONFIGS.keys()]
            baseline_handle = Line2D([0], [0], linestyle='--', color='black', linewidth=2.5, label='Flat Baseline (L=1)')
            condition_handles = [Line2D([0], [0], marker=MARKERS_CONDITIONS[c], color='black', linewidth=0, markersize=8, label=DISPLAY_NAMES_SCAFFOLDS[c]) for c in VALID_SCAFFOLDS]
            
            first_legend = ax.legend(handles=model_handles, loc='upper left', title="Models", frameon=True, fontsize=11)
            ax.add_artist(first_legend)
            ax.legend(handles=[baseline_handle] + condition_handles, loc='upper right', title="Conditions", frameon=True, fontsize=11)
    
    axes[-1].set_xlabel("Length (L)", fontsize=13)
    axes[-1].set_xticks(VALID_L_WITH_BASELINE)
    fig.tight_layout()
    fig.savefig(COMBINED_DIR / "combined_stacked_all_responses.png", dpi=200)
    plt.close(fig)
    print("[STACKED PLOT] Wrote combined_stacked_all_responses.png")

# ---------- Regression ----------
def prep_regression_data(master_df: pd.DataFrame) -> pd.DataFrame:
    df = master_df[master_df["scaffold"].isin(VALID_SCAFFOLDS) & master_df["acc_type"].isin(ACC_ALL)].copy()
    df["scaffold"] = pd.Categorical(df["scaffold"], categories=VALID_SCAFFOLDS, ordered=False)
    df["L"] = df["L"].astype(int)
    present_acc = [c for c in ACC_ALL if c in df["acc_type"].unique()]
    df["acc_type"] = pd.Categorical(df["acc_type"], categories=present_acc, ordered=True)
    print(f"  Loaded: {len(df)} rows, {df['scaffold'].nunique()} scaffolds")
    print(f"  Lengths: {sorted(df['L'].unique())}")
    return df

def add_flat_dummy(df: pd.DataFrame, master_df: pd.DataFrame) -> pd.DataFrame:
    # Get L=1 data and filter to only regression categories (exclude NC)
    l1_data = master_df[master_df["L"] == 1].copy()
    l1_data = l1_data[l1_data["acc_type"].isin(ACC_ALL)]
    
    if l1_data.empty:
        raise ValueError("No L=1 data")
    
    l1_dist = l1_data["acc_type"].value_counts(normalize=True).to_dict()
    print(f"  flat_dummy using L=1 distribution (excluding NC): {l1_dist}")
    
    # Calculate mean timing_s for each length (for flat_dummy rows)
    timing_means = {}
    for L in VALID_L_WITH_BASELINE:
        length_data = df[df["L"] == L]
        if len(length_data) > 0 and "timing_s" in length_data.columns:
            timing_means[L] = length_data["timing_s"].mean()
        else:
            timing_means[L] = 0.0
    
    flat_rows = []
    for L in VALID_L_WITH_BASELINE:
        n_obs = len(df[df["L"] == L])
        mean_timing = timing_means[L]
        for acc_type, prop in l1_dist.items():
            n_samples = int(n_obs * prop)
            for _ in range(n_samples):
                flat_rows.append({
                    "scaffold": "flat_dummy",
                    "L": L,
                    "acc_type": acc_type,
                    "timing_s": mean_timing
                })
    
    flat_df = pd.DataFrame(flat_rows)
    extended_scaffolds = ["flat_dummy"] + VALID_SCAFFOLDS
    flat_df["scaffold"] = pd.Categorical(flat_df["scaffold"], categories=extended_scaffolds, ordered=False)
    flat_df["acc_type"] = pd.Categorical(flat_df["acc_type"], categories=df["acc_type"].cat.categories, ordered=True)
    
    df_with_flat = pd.concat([df, flat_df], ignore_index=True)
    df_with_flat["scaffold"] = pd.Categorical(df_with_flat["scaffold"], categories=extended_scaffolds, ordered=False)
    return df_with_flat

def fit_model(df: pd.DataFrame, label: str):
    df_local = df.copy()
    classes = list(df_local["acc_type"].cat.categories)
    
    present_scaffolds = [s for s in ["flat_dummy"] + VALID_SCAFFOLDS if s in df_local["scaffold"].unique()]
    df_local["scaffold"] = pd.Categorical(df_local["scaffold"], categories=present_scaffolds, ordered=False)
    
def fit_model(df: pd.DataFrame, label: str, include_timing: bool = False):
    """Fit regression model. 
    
    include_timing parameter kept for compatibility but timing model is not run by default.
    Timing-length correlation is documented separately.
    """
    df_local = df.copy()
    classes = list(df_local["acc_type"].cat.categories)
    
    present_scaffolds = [s for s in ["flat_dummy"] + VALID_SCAFFOLDS if s in df_local["scaffold"].unique()]
    df_local["scaffold"] = pd.Categorical(df_local["scaffold"], categories=present_scaffolds, ordered=False)
    
    if len(classes) == 3:
        df_local["acc_type_numeric"] = df_local["acc_type"].cat.codes
        code_mapping = {code: cat for code, cat in enumerate(classes)}
        print(f"  acc_type_numeric mapping: {code_mapping}")
        
        # Choose model based on include_timing flag
        if include_timing:
            formula = "acc_type_numeric ~ C(scaffold) * timing_s"
            model_type = "TIMING"
            suffix = "_timing"
        else:
            formula = "acc_type_numeric ~ C(scaffold) * L"
            model_type = "LENGTH"
            suffix = ""
        
        result = smf.mnlogit(formula, data=df_local).fit(method="newton", maxiter=200, disp=False)
        
        summ_path = REGRESSION_DIR / f"{label}_mnlogit{suffix}_summary.txt"
        with summ_path.open("w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"MULTINOMIAL LOGIT RESULTS ({model_type} MODEL)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"MODEL: {formula}\n\n")
            f.write("NOTE: Length (L) and timing (timing_s) are highly correlated (r ~ 0.9)\n")
            f.write("      Running separate models to avoid multicollinearity.\n\n")
            f.write("OUTCOME VARIABLE CODING:\n")
            for code, cat in code_mapping.items():
                f.write(f"  acc_type_numeric={code} → '{cat}'\n")
            f.write(f"\nREFERENCE CATEGORY: '{classes[0]}' (code=0)\n\n")
            f.write(result.summary().as_text())
        print(f"[MNLogit] {label}: wrote {model_type} model summary")
        return result, classes, True, df_local
    
    elif len(classes) == 2:
        pos = "Incorrect" if "Incorrect" in classes else classes[-1]
        df_local["y_binary"] = (df_local["acc_type"] == pos).astype(int)
        
        if include_timing:
            formula_binary = "y_binary ~ C(scaffold) * timing_s"
            model_type = "TIMING"
            suffix = "_timing"
        else:
            formula_binary = "y_binary ~ C(scaffold) * L"
            model_type = "LENGTH"
            suffix = ""
        
        result = smf.logit(formula_binary, data=df_local).fit(method="newton", maxiter=200, disp=False)
        
        summ_path = REGRESSION_DIR / f"{label}_logit{suffix}_summary.txt"
        with summ_path.open("w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"BINARY LOGIT RESULTS ({model_type} MODEL)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"MODEL: {formula_binary}\n\n")
            f.write("NOTE: Length (L) and timing (timing_s) are highly correlated (r ~ 0.9)\n")
            f.write("      Running separate models to avoid multicollinearity.\n\n")
            f.write(result.summary().as_text())
        print(f"[Logit] {label}: wrote {model_type} model summary")
        return result, classes, False, df_local
    
    raise RuntimeError(f"{label}: only one class present")

def analyze_timing_length_correlation(master_dfs: dict):
    """Analyze correlation between timing_s and L for each model."""
    print("\n" + "="*60)
    print("TIMING vs LENGTH CORRELATION ANALYSIS")
    print("="*60)
    
    results = {}
    for model_name, df in master_dfs.items():
        if df.empty:
            continue
        
        # Filter to experimental scaffolds only
        analysis_df = df[df["scaffold"].isin(VALID_SCAFFOLDS)].copy()
        
        if "timing_s" not in analysis_df.columns or len(analysis_df) < 10:
            print(f"\n{model_name}: Insufficient data")
            continue
        
        # Calculate Pearson correlation
        correlation = analysis_df[["L", "timing_s"]].corr().iloc[0, 1]
        
        # Calculate correlation by scaffold
        scaffold_corrs = {}
        for scaffold in VALID_SCAFFOLDS:
            scaffold_df = analysis_df[analysis_df["scaffold"] == scaffold]
            if len(scaffold_df) > 5:
                scaffold_corr = scaffold_df[["L", "timing_s"]].corr().iloc[0, 1]
                scaffold_corrs[scaffold] = scaffold_corr
        
        results[model_name] = {
            "overall": correlation,
            "by_scaffold": scaffold_corrs
        }
        
        print(f"\n{model_name}:")
        print(f"  Overall correlation (L vs timing_s): {correlation:.3f}")
        print(f"  By scaffold:")
        for scaffold, corr in scaffold_corrs.items():
            print(f"    {scaffold}: {corr:.3f}")
    
    # Save to file
    if results:
        output_file = REGRESSION_DIR / "timing_length_correlation.txt"
        with output_file.open("w", encoding="utf-8") as f:
            f.write("TIMING vs LENGTH CORRELATION ANALYSIS\n")
            f.write("="*80 + "\n\n")
            f.write("Pearson correlation between conversation length (L) and response time (timing_s)\n")
            f.write("High correlation may indicate multicollinearity in regression models.\n\n")
            
            for model_name, corr_data in results.items():
                f.write(f"{model_name}:\n")
                f.write(f"  Overall: {corr_data['overall']:.3f}\n")
                f.write(f"  By scaffold:\n")
                for scaffold, corr in corr_data['by_scaffold'].items():
                    f.write(f"    {scaffold}: {corr:.3f}\n")
                f.write("\n")
        
        print(f"\n[SAVED] {output_file}")
    
    return results

def run_regression_analysis(master_df: pd.DataFrame, label: str):
    try:
        df_prep = prep_regression_data(master_df)
        if df_prep.empty:
            print(f"[WARN] {label}: No data for regression")
            return
        
        df_with_flat = add_flat_dummy(df_prep, master_df)
        
        # Length model only (timing correlation documented separately)
        print(f"  [LENGTH MODEL] Fitting C(scaffold) * L...")
        result_length, classes, is_mn, df_fitted = fit_model(df_with_flat, label, include_timing=False)
        
        print(f"[DONE] {label}: regression complete")
        
    except Exception as e:
        print(f"[ERROR] {label}: {e}")
        import traceback
        traceback.print_exc()

def calculate_nc_statistics(master_dfs: dict):
    """Calculate NC percentages by model, scaffold, and length."""
    print("\n" + "="*60)
    print("NC (NON-COMPLIANT) RESPONSE STATISTICS")
    print("="*60)
    
    nc_stats = {}
    
    for model_name, df in master_dfs.items():
        if df.empty:
            continue
        
        model_stats = {
            "overall": {},
            "by_scaffold": {}
        }
        
        # Overall NC rate by length
        for L in VALID_L_WITH_BASELINE:
            length_df = df[df["L"] == L]
            if len(length_df) > 0:
                nc_count = len(length_df[length_df["acc_type"] == "NC"])
                total = len(length_df)
                pct = (nc_count / total * 100) if total > 0 else 0
                model_stats["overall"][L] = {"nc_count": nc_count, "total": total, "pct": pct}
        
        # NC rate by scaffold and length
        for scaffold in VALID_SCAFFOLDS:
            scaffold_df = df[df["scaffold"] == scaffold]
            if len(scaffold_df) == 0:
                continue
            
            scaffold_stats = {}
            for L in VALID_L_WITH_BASELINE:
                length_df = scaffold_df[scaffold_df["L"] == L]
                if len(length_df) > 0:
                    nc_count = len(length_df[length_df["acc_type"] == "NC"])
                    total = len(length_df)
                    pct = (nc_count / total * 100) if total > 0 else 0
                    scaffold_stats[L] = {"nc_count": nc_count, "total": total, "pct": pct}
            
            if scaffold_stats:
                model_stats["by_scaffold"][scaffold] = scaffold_stats
        
        nc_stats[model_name] = model_stats
        
        # Print summary
        print(f"\n{model_name}:")
        print(f"  Overall NC rates by length:")
        for L in VALID_L_WITH_BASELINE:
            if L in model_stats["overall"]:
                stats = model_stats["overall"][L]
                print(f"    L={L}: {stats['pct']:.1f}% ({stats['nc_count']}/{stats['total']})")
        
        print(f"  NC rates by scaffold:")
        for scaffold in VALID_SCAFFOLDS:
            if scaffold in model_stats["by_scaffold"]:
                print(f"    {scaffold}:")
                scaffold_stats = model_stats["by_scaffold"][scaffold]
                for L in VALID_L_WITH_BASELINE:
                    if L in scaffold_stats:
                        stats = scaffold_stats[L]
                        print(f"      L={L}: {stats['pct']:.1f}% ({stats['nc_count']}/{stats['total']})")
    
    # Save to file
    if nc_stats:
        output_file = RUNS_DIR / "nc_statistics.txt"
        with output_file.open("w", encoding="utf-8") as f:
            f.write("NON-COMPLIANT (NC) RESPONSE STATISTICS\n")
            f.write("="*80 + "\n\n")
            f.write("NC responses are those that could not be parsed as YES, NO, or IDK.\n")
            f.write("This includes template leakage, malformed responses, and refusals.\n\n")
            
            for model_name, model_stats in nc_stats.items():
                f.write(f"{model_name}:\n")
                f.write(f"  Overall NC rates by length:\n")
                for L in VALID_L_WITH_BASELINE:
                    if L in model_stats["overall"]:
                        stats = model_stats["overall"][L]
                        f.write(f"    L={L}: {stats['pct']:.1f}% ({stats['nc_count']}/{stats['total']})\n")
                
                f.write(f"\n  NC rates by scaffold:\n")
                for scaffold in VALID_SCAFFOLDS:
                    if scaffold in model_stats["by_scaffold"]:
                        f.write(f"    {scaffold}:\n")
                        scaffold_stats = model_stats["by_scaffold"][scaffold]
                        for L in VALID_L_WITH_BASELINE:
                            if L in scaffold_stats:
                                stats = scaffold_stats[L]
                                f.write(f"      L={L}: {stats['pct']:.1f}% ({stats['nc_count']}/{stats['total']})\n")
                f.write("\n")
        
        print(f"\n[SAVED] {output_file}")
    
    return nc_stats

def calculate_baseline_stats(master_dfs: dict):
    print("\n" + "="*60)
    print("BASELINE (L=1) RESPONSE PERCENTAGES")
    print("="*60)
    
    results = {}
    for model_name, df in master_dfs.items():
        if df.empty:
            continue
        
        baseline = df[df["L"] == 1].copy()
        if baseline.empty:
            continue
        
        total = len(baseline)
        percentages = {}
        for acc_type in ACC_ALL_WITH_NC:
            count = len(baseline[baseline["acc_type"] == acc_type])
            percentages[acc_type] = (count / total) * 100 if total > 0 else 0
        
        results[model_name] = percentages
        print(f"\n{model_name} (n={total}):")
        for acc_type in ACC_ALL_WITH_NC:
            print(f"  {acc_type}: {percentages[acc_type]:.1f}%")
    
    if results:
        output_file = RUNS_DIR / "baseline_percentages.txt"
        with output_file.open("w", encoding="utf-8") as f:
            f.write("BASELINE (L=1) RESPONSE PERCENTAGES\n")
            f.write("="*80 + "\n\n")
            for model_name, percentages in results.items():
                f.write(f"{model_name}:\n")
                for acc_type in ACC_ALL_WITH_NC:
                    f.write(f"  {acc_type}: {percentages[acc_type]:.1f}%\n")
                f.write("\n")
        print(f"\n[SAVED] {output_file}")
    
    return results

# ---------- Main ----------
def main():
    print("="*60)
    print("ANALYSIS SCRIPT")
    print("="*60)
    
    keep_ids, gold_map = load_final_ids_and_gold(FINAL_JSONL)
    print(f"Loaded {len(keep_ids)} BoolQ IDs with gold labels from final dataset\n")
    
    # Step 1: Parse conversations and build master dataframes
    print("="*60)
    print("STEP 1: Parsing Conversations & Creating detailed_rows.csv")
    print("="*60)
    
    master_dfs = {}
    for model_name, config in MODEL_CONFIGS.items():
        master_df = build_master_dataframe(model_name, config["folder"], keep_ids, gold_map)
        master_dfs[model_name] = master_df
    
    # Step 2: Calculate baseline stats
    print("\n" + "="*60)
    print("STEP 2: Baseline Statistics")
    print("="*60)
    calculate_baseline_stats(master_dfs)
    
    # Step 2b: Calculate NC statistics
    print("\n" + "="*60)
    print("STEP 2b: NC Statistics")
    print("="*60)
    calculate_nc_statistics(master_dfs)
    
    # Step 3: Create combined plots
    print("\n" + "="*60)
    print("STEP 3: Creating Combined Plots")
    print("="*60)
    
    flat_dummies = {}
    for model_name, df in master_dfs.items():
        if not df.empty:
            flat_dummy_df = calculate_flat_dummy_for_viz(df)
            if not flat_dummy_df.empty:
                flat_dummies[model_name] = flat_dummy_df
    
    for response_type in ["Correct", "Incorrect", "IDK"]:
        plot_combined_by_response_type(master_dfs, response_type, flat_dummies)
    
    plot_stacked_combined(master_dfs, flat_dummies)
    
    # Create per-model plots (all outcomes on one panel per model)
    print("\n[CREATING PER-MODEL PLOTS]")
    for model_name, master_df in master_dfs.items():
        if not master_df.empty:
            plot_per_model_all_outcomes(model_name, master_df)
    
    # Step 4: Timing-Length Correlation Analysis
    print("\n" + "="*60)
    print("STEP 4: Timing-Length Correlation Analysis")
    print("="*60)
    analyze_timing_length_correlation(master_dfs)
    
    # Step 5: Run regressions
    print("\n" + "="*60)
    print("STEP 5: Running Regression Analysis")
    print("="*60)
    
    for model_name, master_df in master_dfs.items():
        if not master_df.empty:
            print(f"\n--- Regression for {model_name} ---")
            run_regression_analysis(master_df, model_name)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  - detailed_rows.csv files: runs/MODEL/SCAFFOLD/")
    print(f"  - Combined plots: {COMBINED_DIR}")
    print(f"  - Regression results: {REGRESSION_DIR}")

if __name__ == "__main__":
    main()
