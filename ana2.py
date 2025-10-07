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
    # 1. levensein
    from rapidfuzz.distance import Levenshtein
    from itertools import combinations
    from tqdm import tqdm
    from collections import Counter

    # from nltk.translate.bleu_score import sentence_bleu

    # from nltk.corpus import stopwords
    # import nltk
    # nltk.download('stopwords')
    # from nltk.tokenize import word_tokenize

    # text = "This is a simple example to demonstrate stop word filtering."
    # stop_words = set(stopwords.words('english'))
    # words = word_tokenize(text)

    # filtered = [w for w in words if w.lower() not in stop_words]
    # print(filtered)

    fpth = Path("/disk/lani/karl/data/boolq_enriched.jsonl")
    fout = Path("/disk/lani/karl/data/boolq_levenshetin_sim.jsonl")

    # 
    threshold = 0.5

    df = pd.read_json(fpth, lines=True)
    print(df.shape)
    # df2 = pd.read_csv(fin)
    drop = {}
    i1s, i2s, ss, s1s, s2s = [], [], [], [], []
    # print(df.shape, df2.shape)
    for (domain, this_df) in df.groupby("domain"):
        for ix, row in this_df.iterrows():
            i1 = row["id"]
            s1 = row.get("corrected") or row.get("question")
            for iy, row in this_df.iterrows():
                if iy <= ix:
                    continue
            
                i2 = row["id"]
                s2 = row.get("corrected") or row.get("question")
                sim = Levenshtein.normalized_similarity(s1.strip().lower(), s2.strip().lower())

                i1s.append(i1)
                i2s.append(i2)
                s1s.append(s1)
                s2s.append(s2)
                ss.append(sim)
    ndf = pd.DataFrame()
    ndf["id 1"] = i1s
    ndf["id 2"] = i2s
    ndf["sim"] = ss
    ndf["s1"] = s1s
    ndf["s2"] = s2s

    breakpoint()
    ndf.to_csv(fout, lines=True, orient="records")



        
        # print(domain, this_df.shape[0])
    #     for index, row in this_df.iterrows():
    #         if drop.get(index):
    #             continue
    #         that = df2[(df2["index 1"] == index) | (df2["index 2"] == index)]
    #         that = that[that["score"] >= threshold]
    #         for each in that["index 1"].tolist() + that["index 2"].tolist():
    #             if each == index:
    #                 continue
    #             that_id = df.loc[each, "id"]
    #             drop[each] = drop.get(each, []) +  [that_id]
    #             breakpoint()
    # col = []
    # print(len(drop))
    # for index, row in df.iterrows():
    #     va = drop.get(index, [])
    #     if not va:
    #         continue
    #     breakpoint()
    #     col.append(", ".join(tmp))
    # df["drop"] = col

    # df.to_json(Path("/disk/lani/karl/data/boolq_drop.jsonl"), lines=True, orient="records")

    # ids2 = df2[df2["score"] >=0.8]["index 1"].tolist() + df2[df2["score"] >=0.8]["index 2"].tolist()
    # count = Counter(ids2)
    # big_drop = 
    # breakpoint()


    #     combs = list(combinations(this_df["id"].tolist(), 2))
    #     for i1, i2 in combs:

    #         s1 = this_df[this_df["id"] == i1]["corrected"].item() or this_df[this_df["id"] == i1]["question"].item()
    #         s2 = this_df[this_df["id"] == i2]["corrected"].item() or this_df[this_df["id"] == i2]["question"].item()

    #         s1 = s1.lower()
    #         s2 = s2.lower()
    #         sim = Levenshtein.normalized_similarity(s1, s2)
    #         breakpoint()

    # breakpoint()
    # domains = df.domain.unique()
    # for this_domain in domains:
    #     this_df = df[df["domain"] == ]
   
    # out = {
    #     "index 1": [],
    #     "index 2": [],
    #     "question 1": [],
    #     "question 2": [],
    #     "score": []
    # }
    # df = pd.read_csv(fin)
    # ids = df["index 1"].tolist() + df["index 2"].tolist()
    # ids = list(set(ids))
    # # print(len(ids))
    # # print(len(set(ids)))
    # # 去除完全一致的，然后从sim高中的随机drop
    # breakpoint()
    # threshold = 0.8
    # df2 = df[df["score"] < threshold]   # 这部分是可以保留的，可能还是有的最后会被去除
    
    # uniq = []
    # for _id in ids:
    #     this = df[(df["index 1"] == _id) or (df["index 2"] == _id)]
    #     this = this[this["score"] < threshold]
    #     this["index 1"].tolist() + this["index 2"].tolist()

    # df = 
    # for _, row in df.iterrows():
    #     i1, i2, s1, s2, scr = row
    #     if scr > 0.5:
    #         print(s1)
    #         print(s2)
    #         print(scr)
    #         print("-_______________")
    # breakpoint()
    #     ss1 = " ".join([s for s in s1.split() if s not in stop_words])
    #     ss2 = " ".join([s for s in s2.split() if s not in stop_words])
    #     scr = Levenshtein.normalized_similarity(ss1, ss2)

    #     out["index 1"].append(i1)
    #     out["index 2"].append(i2)
    #     out["question 1"].append(s1)
    #     out["question 2"].append(s2)
    #     out["score"].append(scr)
    # df =  pd.DataFrame(out)
    # df.to_csv(fout, index=False)


    # df = pd.read_csv(fin)
    # out = {
    #     "index 1": [],
    #     "index 2": [],
    #     "question 1": [],
    #     "question 2": [],
    #     "score": []
    # }
    # for _, row in tqdm(df.iterrows(), total=len(df)):
    #     i1, i2, s1, s2, _ = row
    #     scr = sentence_bleu([s1.lower().split()], s2.lower().split())
    #     out["index 1"].append(i1)
    #     out["index 2"].append(i2)
    #     out["question 1"].append(s1)
    #     out["question 2"].append(s2)
    #     out["score"].append(scr)
    # df =  pd.DataFrame(out)
    # df.to_csv(fout, index=False)


    #         res.append((i1, i2, nd))
    # res.sort(key=lambda x:x[-1])
    # out = {
    #     "index 1": [],
    #     "index 2": [],
    #     "question 1": [],
    #     "question 2": [],
    #     "score": []
    # }
    # for (i1, i2, scr) in res:
    #     s1 = df.iloc[i1]["corrected"] or df.iloc[i1]["question"]
    #     s2 = df.iloc[i2]["corrected"] or df.iloc[i2]["question"]

    #     out["index 1"].append(i1)
    #     out["index 2"].append(i2)
    #     out["question 1"].append(s1)
    #     out["question 2"].append(s2)
    #     out["score"].append(scr)

    # df = pd.DataFrame(out)
    # df.to_csv(fout, index=False)

        
    

