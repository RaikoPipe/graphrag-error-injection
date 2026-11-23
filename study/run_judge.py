"""Judge all unique (question, gold, answer) triples across result files with the primary judge."""
import sys, os, glob
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
os.environ.setdefault("LLM_CACHE", "cache/llm_cache.sqlite")
import pandas as pd, qa
JUDGE = sys.argv[1] if len(sys.argv) > 1 else "ollama:deepseek-v4.1-flash"
W = int(sys.argv[2]) if len(sys.argv) > 2 else 8
fs = [f for f in glob.glob("results/gen_*.parquet")]
d = pd.concat([pd.read_parquet(f, columns=["question", "gold", "answer"]) for f in fs]).drop_duplicates()
items = list(d.itertuples(index=False, name=None))
print("unique items", len(items), flush=True)
lab = qa.judge(JUDGE, items, workers=W)
out = pd.DataFrame(items, columns=["question", "gold", "answer"]); out["correct"] = lab
tag = JUDGE.replace(":", "_")
out.to_parquet(f"results/judgments_{tag}.parquet"); print("DONE", out.correct.isna().sum(), out.correct.mean())
