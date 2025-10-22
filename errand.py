import pandas as pd
from pathlib import Path
import json

from tqdm import tqdm

#! 1， 修改manifest，manifest作为索引，
#! 2. 修改输出

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
#? 处理manifest文件，一个模型同一个scaffold下应该只有一个
# mpth = Path("/disk/lani/karl/runs/baseline_phi3/_manifest.json")
# mani = read_json(mpth)
# scaff, lengths = mani.pop("scaffold"), mani.pop("lengths")

# # 'mode', 'scaffold', 'lengths', 'model', 'device', 'dtype', 'decoding', 'assets_root', 'items
# # _ = mani.pop("items")
# items = {}
# for each in mani.pop("items"):
#     _id = each.pop("boolq_id")
#     length = each.pop("length")
#     flist = items.get(_id, [])
#     items[_id] = flist
#     flist.append("L{}_{}.json".format(length, scaff))

# mani["files"] = items
# fout = Path("/disk/lani/karl/runs/phi3/").joinpath("_manifest.json")
# write_json(fout, mani)

# exit()

#? 根据输出文件生成manifest文件
# mpth = Path("/disk/lani/karl/runs/phi3/_manifest.json")
# mani = read_json(mpth)
# mani_items = mani["items"]

# files = Path("/disk/lani/karl/runs/phi3/").glob("L*meta.json")
# for f in files:
#     df = read_json(f)
#     this_items = df["items"]
#     for each in this_items:
#         _id = each["boolq_id"]
#         mani_items[_id] = mani_items.get(_id, []) + [f.name]

# write_json(mpth, mani)

# exit()


# #? 处理单个文件
# _dir = Path("/disk/lani/karl/runs/meta_phi3")
# out_dir = Path("/disk/lani/karl/runs/phi3/")
# files = _dir.glob("*_L21_*.json") # 6,11,16,21

# out = {
#     "mode": "length",
#     "scaffold": None,
#     "length": None,
#     # "boolq_id": boolq_id,
#     # "question": q_corr,
#     # "gold_answer": erow.get("answer"),
#     # "enriched": {"topic_primary": erow.get("topic_primary"),
#     #                 "topic_related": erow.get("topic_related") or []},
#     #! + domain
#     "model": None,
#     "decoding": None,
#     # "timing_s": round(elapsed, 3),
#     # "transcript": transcript,
# }

# # 整合输出文件
# ini = True
# items = []
# for f in files:
#     if "manifest" in f.name:
#         continue
#     rows = read_json(f)
#     if ini:
#         for k, v in out.items():
#             out[k] = rows[k]
#         ini = False

#     for k in out.keys():
#         rows.pop(k)
#     # 修改transcript
#     transcript = []
#     _tr = rows.pop("transcript")
#     assert not len(_tr) % 2
#     _tmp = { }
#     for _, each in enumerate(_tr):
#         _tmp["turn"] = each.pop("turn")
#         _tmp["user"] = each.get("prompt", _tmp.get("user"))
#         _tmp["model"] = each.get("reply", _tmp.get("model"))

#         if not (_ + 1) % 2:
#             assert  _tmp["model"] and _tmp["user"]
#             transcript.append(_tmp)
#             _tmp = {}
#     rows["transcript"] = transcript
#     items.append(rows)
# out["items"] = items

        
# fout = out_dir.joinpath("L{}_{}.json".format(out.get("length", 0), out.get("scaffold", 0)))   
# write_json(fout, out)

# fpth = Path("/disk/lani/karl/runs/phi3/L11_meta.json")
# df = read_json(fpth)
# items = []
# for each in df["items"]:
#     if "model" in each or "decoding" in each:
#         each.pop("model")
#         each.pop("decoding")
#     items.append(each)
# df["items"] = items 
# write_json(Path("/disk/lani/karl/runs/phi3/L11_meta_cor.json"), df)

#? 检查L11
# fpth = Path("/disk/lani/karl/runs/phi3/L11_meta.json")
# df = read_json(fpth)
# items = df["items"]
# memo = {}
# for each in items:
#     _id = each.pop("boolq_id")
#     memo[_id] = memo.get(_id, 0) + 1
# for k, v in memo.items():
#     if v > 1:
#         print(k)


#? 专程ndjson文件，可以新增行
# manifest也改成ndjson文件。。， 好像不用，
# import json


#! 看运行时间
_dir = Path("/disk/lani/karl/runs/phi4/")
_all = 0 
for fd in _dir.glob("*"):

    if fd.name not in ["baseline", "semantic", "meta", "misleading", "underspecified"]:
        continue
    print(">>: ", fd.name)
    time = 0
    for f in fd.glob("L*"):
        d = read_jsonl(f)

        for each in d[1:]:  # skip common 
            try:
                time += each["elapsed_s"] 
            except:
                time += each["timing_s"]
    _all += time
    print(time)
print(_all)

# for f in Path("/disk/lani/karl/runs/phi3_cp2/").glob("L*.json"):
#     # fin = Path("/disk/lani/karl/runs/phi3/L21_meta.json")
#     fin = f
#     fout = fin
#     # fout = Path("/disk/lani/karl/runs/phi3/").joinpath("{}.ndjson".format(f.stem))
#     # fout = Path("/disk/lani/karl/runs/phi3/L21_meta.ndjson")

#     data = read_json(fin)

#     items = data.pop("items")

#     # write setting 
#     with open(fout, "w") as f:
#         f.write(json.dumps([data]) + "\n")

#     # add items 
#     # with open(fout, "a") as f:
#     #     for item in items:
#     #         f.write(json.dumps(item) + "\n")
#     with open(fout, "a") as f:
#         f.writelines(json.dumps(item) + "\n" for item in items)
#     # 加载ndjson
#     df = read_jsonl(fout)

