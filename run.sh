
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
python run_length.py \
--scaffold meta \
--lengths 6,11,16,21 \
--in_enriched /disk/lani/karl/data/boolq_enriched.jsonl \
--out_root /disk/lani/karl/runs/phi3 \
--assets_root /disk/lani/karl/data/length \
--model microsoft/Phi-3-mini-4k-instruct \
--device cuda \
--dtype float32 
# If scaffold is light or rich, lengths can be multiple values but not 1
# python run_length.py \
# --scaffold rich \
# --lengths 11,21,31,41 \
# --in_dev /disk/lani/karl/data/dev.jsonl \
# --in_enriched /disk/lani/karl/data/boolq_enriched.jsonl \
# --out_root /disk/lani/karl/runs/phi3 \
# --assets_root /disk/lani/karl/data/length \
# --skip_existing \
# --model microsoft/Phi-3-mini-4k-instruct \
# --device cuda \
# --dtype float16 \
# --max_new_tokens_tasks 64 \
# --max_new_tokens_final 96 \
# --temperature 0 \
# --stop-seq "### User" \
# Example to run only a few specific IDs:
#     --id-include dev_0003-dev_0020 
#     --skip-existing \
# or a larger range:
    # --num 100 \




