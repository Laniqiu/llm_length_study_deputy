
# run_length.py — Length vs Veracity (BoolQ) runner, rolling context only.

# Assets:
#   data/length/baseline/L{L}.jsonl          # each line: {"length":L,"turn":k,"prompt":"..."}
#   data/length/light/L{L}_info-light.jsonl   # each line: {"length":L,"scaffold":"light","turn":k,"prompt":"..."}
#   data/length/rich/L{L}_info-rich.jsonl     # each line: {"length":L,"scaffold":"rich","turn":k,"prompt":"..."}

# Usage: change scaffold and out-root as needed
  # Baseline (L=1 only)
# python run_length.py \
#   --scaffold baseline \
#   --lengths 1 \
#   --in_enriched /disk/lani/karl/data/boolq_enriched.jsonl \
#   --out_root /disk/lani/karl/runs/baseline_phi3 \
#   --assets_root /disk/lani/karl/data/length \
#   --model microsoft/Phi-3-mini-4k-instruct \
#   --device cuda \
#   --dtype float32 \
#   --max_new_tokens_tasks 64 \
#   --max_new_tokens_final 96 \
#   --temperature 0 \
#   --stop_seq "### User"

# Meta, any subset of {6,11,16,21}
# python run_length.py \
# --scaffold meta \
# --lengths 6,11,16,21 \
# --in_enriched /disk/lani/karl/data/boolq_enriched.jsonl \
# --out_root /disk/lani/karl/runs/phi3 \
# --assets_root /disk/lani/karl/data/length \
# --model microsoft/Phi-3-mini-4k-instruct \
# --device cuda \
# --dtype float32 
# Semantic
python run_length.py \
  --scaffold misleading \
  --lengths 6\
  --in_enriched /disk/lani/karl/data/boolq_final.jsonl \
  --out_root /disk/lani/karl/runs/phi3 \
  --assets_root /disk/lani/karl/data/length \
  --model microsoft/Phi-3-mini-4k-instruct \
  --device cuda \
  --dtype float32 






