# Set environment variable
export PROJECT_ROOT="/disk/lani/karl/"
echo "PROJECT_ROOT: $PROJECT_ROOT"


python analyze_length.py \
    --in_root /disk/lani/karl/runs/phi4/baseline \
    --out_dir  /disk/lani/karl/runs/phi4/analysis/baseline\
    --make_plots


python analyze_length.py \
    --in_root /disk/lani/karl/runs/phi4/meta \
    --out_dir  /disk/lani/karl/runs/phi4/analysis/meta\
    --make_plots


python analyze_length.py \
    --in_root /disk/lani/karl/runs/phi4/semantic \
    --out_dir  /disk/lani/karl/runs/phi4/analysis/semantic\
    --make_plots

python analyze_length.py \
    --in_root /disk/lani/karl/runs/phi4/misleading \
    --out_dir  /disk/lani/karl/runs/phi4/analysis/misleading\
    --make_plots

python analyze_length.py \
    --in_root /disk/lani/karl/runs/phi4/underspecified \
    --out_dir  /disk/lani/karl/runs/phi4/analysis/underspecified\
    --make_plots