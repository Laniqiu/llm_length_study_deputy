
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
#   --scaffold meta \
#   --lengths 6 \
#   --in_final /disk/lani/karl/data/boolq_final.jsonl \
#   --out_root /disk/lani/karl/runs/phi4/meta \
#   --model phi4-mini \
#   --device cuda \
#   --dtype float32

# python run_length.py \
#   --scaffold semantic \
#   --lengths 6 \
#   --in_final /disk/lani/karl/data/boolq_final.jsonl \
#   --out_root /disk/lani/karl/runs/phi4/semantic \
#   --model phi4-mini \
#   --device cuda \
#   --dtype float32
python run_length.py \
  --scaffold underspecified \
  --lengths 6 \
  --in_final /disk/lani/karl/data/boolq_final.jsonl \
  --out_root /disk/lani/karl/runs/phi4/underspecified \
  --model phi4-mini \
  --device cuda \
  --dtype float32

  python run_length.py \
  --scaffold misleading \
  --lengths 6 \
  --in_final /disk/lani/karl/data/boolq_final.jsonl \
  --out_root /disk/lani/karl/runs/phi4/misleading \
  --model phi4-mini \
  --device cuda \
  --dtype float32