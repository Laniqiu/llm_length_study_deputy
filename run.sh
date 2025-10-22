
# python run_length.py \
#   --scaffold baseline \
#   --lengths 1 \
#   --in_final /disk/lani/karl/data/boolq_final.jsonl \
#   --out_root /disk/lani/karl/runs/phi4/baseline \
#   --model phi4-mini \
#   --device cuda \
#   --dtype float32


export PROJECT_ROOT="/disk/lani/karl/"
echo "PROJECT_ROOT: $PROJECT_ROOT"

# python run_length.py \
#   --scaffold misleading \
#   --lengths 6 \
#   --in_final /disk/lani/karl/data/boolq_final.jsonl \
#   --out_root /disk/lani/karl/runs/phi3/misleading \
#   --model phi3-mini \
#   --device cuda \
#   --dtype float32

python run_length.py \
  --scaffold semantic \
  --lengths 11,16,21 \
  --in_final /disk/lani/karl/data/boolq_final.jsonl \
  --out_root /disk/lani/karl/runs/phi4/semantic \
  --out_root /disk/lani/karl/runs/phi4/semantic \
  --model phi4-mini \
  --device cuda \
  --dtype float32
