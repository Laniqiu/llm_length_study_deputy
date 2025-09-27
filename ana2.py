import pandas as pd
from pathlib import Path
import json

from tqdm import tqdm


def read_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
        
def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows



if __name__ == "__main__":
    #! add domain to output file
    # fpth = Path("/disk/lani/karl/runs/phi3/L1_baseline.json")
    # bpth = Path("/disk/lani/karl/data/boolq_enriched.jsonl")


    # out_dir = Path("/disk/lani/karl/runs/phi3_domain")
    # out_dir.mkdir(exist_ok=True)
    # fout = out_dir.joinpath(fpth.name)

    # data = read_jsonl(fpth)
    # ori = read_jsonl(bpth)
    # bq = pd.read_json(bpth, lines=True)

    # for ix, each in enumerate(data):
    #     if not ix:  # 直接写出
    #         with open(fout, "w") as f:
    #             f.write(json.dumps(each) +"\n")

    #     else:  # 写出headline
    #         domain = bq[bq["id"]==each["boolq_id"]]["domain"].item()
    #         each["domain"] = domain
    #         with open(fout, "a") as f:
    #             f.write(json.dumps(each) + "\n")

    #! check sim between qs
    fpth = Path("/disk/lani/karl/data/boolq_enriched.jsonl")

    df = pd.read_json(fpth, lines=True)
    domains = df.domain.unique()

    for each in 



    

# 加入domain信息