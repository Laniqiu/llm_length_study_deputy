export PROJECT_ROOT="/disk/lani/karl/"
echo "PROJECT_ROOT: $PROJECT_ROOT"


python analysis_agg.py \
  --in_dir /disk/lani/karl/runs/phi3/analysis \
  --in_final /disk/lani/karl/data/boolq_final.jsonl