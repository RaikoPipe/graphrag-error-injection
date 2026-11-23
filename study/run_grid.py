"""Usage: python run_grid.py <model> <grid: main|sub|sub03> <workers> [retrieval_only]"""
import sys, os, json, time
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
os.environ.setdefault("LLM_CACHE", "cache/llm_cache.sqlite")
import pandas as pd, kg, runner

model, grid, workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
Q = json.load(open("questions.json")); paras = json.load(open("paragraphs.json"))
trip = json.load(open("extract_gptoss120b_all.json"))
edges, _ = kg.build_graph(Q, paras, trip)
RATES = {"main": [0.1, 0.2, 0.3, 0.5], "sub": [0.1, 0.3], "sub03": [0.3]}[grid]
RETR = ("KHOP", "PPR") if grid == "main" else ("KHOP",)
conds = [("CLEAN", 0.0)] + [(op, r) for op in kg.OPS for r in RATES]
os.makedirs("results", exist_ok=True)
tag = model.replace(":", "_").replace("/", "_")
outp = f"results/gen_{tag}_{grid}.parquet"
allrows = []
t0 = time.time()
for op, r in conds:
    rows = runner.condition_rows(edges, Q, "EDGE_DEL" if op == "CLEAN" else op, r, 0, retrievers=RETR)
    if op == "CLEAN":
        for x in rows: x["op"] = "CLEAN"
    out = runner.generate(rows, model, workers=workers)
    allrows += out
    pd.DataFrame(allrows).to_parquet(outp)
    print(op, r, len(out), "errs", sum(1 for o in out if o["err"]), round(time.time() - t0), flush=True)
print("DONE", len(allrows), round(time.time() - t0))
