import os, sys, json
sys.path.insert(0, ".")
import pandas as pd, kg
Q = json.load(open("questions.json")); paras = json.load(open("paragraphs.json"))
edges, _ = kg.build_graph(Q, paras, json.load(open("extract_gptoss120b_all.json")))
recs = []
conds = [("CLEAN", 0.0)] + [(op, r) for op in kg.OPS for r in (0.1, 0.2, 0.3, 0.5)]
for op, r in conds:
    E, log = kg.inject(edges, "EDGE_DEL" if op == "CLEAN" else op, r, 0); idx = kg.Index(E)
    for R in ("KHOP", "PPR"):
        for q in Q:
            ranked, seeds = kg.RETRIEVERS[R](idx, q["question"])
            hit = set(); corrupt_hit = 0
            for i in ranked:
                e = idx.edges[i]
                for (qq, k) in e["gold"]:
                    if qq == q["id"]:
                        hit.add(k); corrupt_hit += e["corrupt"]
            n = len(q["evidences"])
            recs.append(dict(qid=q["id"], op=op, rate=r, retriever=R, present_all=len(hit) == n, present_frac=len(hit) / n,
                             n_corrupt_retrieved=corrupt_hit, linked=len(seeds) > 0))
    print(op, r, flush=True)
pd.DataFrame(recs).to_parquet("results/presence.parquet"); print("DONE")
