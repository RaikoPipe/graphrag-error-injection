"""Deterministic retrieval-level grid: all ops x rates x 3 seeds x 2 retrievers (no LLM)."""
import sys, os, json, time
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import pandas as pd, kg, runner
Q = json.load(open("questions.json")); paras = json.load(open("paragraphs.json"))
edges, _ = kg.build_graph(Q, paras, json.load(open("extract_gptoss120b_all.json")))
recs = []
for seed in (0, 1, 2):
    for op in kg.OPS:
        for r in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
            if r == 0 and (op != "EDGE_DEL" or seed): continue
            rows = runner.condition_rows(edges, Q, op, r, seed, provs=(False,))
            for x in rows:
                x.pop("prompt"); x["false_vals"] = len(x["false_vals"])
                if r == 0: x["op"] = "CLEAN"
            recs += rows
            print(seed, op, r, flush=True)
pd.DataFrame(recs).to_parquet("results/retrieval_grid.parquet"); print("DONE", len(recs))
