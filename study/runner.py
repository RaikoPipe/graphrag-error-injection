"""Builds condition rows (retrieval + prompts) and runs generation/judging."""
import json, hashlib, collections
import numpy as np, pandas as pd
import kg, qa, llm

_GI = {}
def gold_index(edges):
    k = id(edges)
    if k not in _GI:
        d = collections.defaultdict(lambda: (set(), set()))
        for e in edges:
            for (q, _) in e["gold"]:
                d[q][0].add(e["id"]); d[q][1].update([e["h"], e["t"]])
        _GI[k] = d
    return _GI[k]

def condition_rows(edges, Q, op, rate, seed, retrievers=("KHOP", "PPR"), provs=(False, True), B=30):
    E, log = kg.inject(edges, op, rate, seed)
    idx = kg.Index(E)
    rows = []
    for R in retrievers:
        for q in Q:
            ranked, seeds = kg.RETRIEVERS[R](idx, q["question"], B=B)
            frac, allsup = kg.gold_support(idx, ranked, q["id"], len(q["evidences"]))
            gids, gnodes = gold_index(edges)[q["id"]]
            touched = bool(gnodes & log["touched_nodes"]) or bool(gids & log["touched_edges"])
            # injected false values present in context (for adoption scoring)
            false_vals = set()
            for i in ranked:
                e = idx.edges[i]
                if e["corrupt"] or e["spurious"]: false_vals.add(e["t"])
                if op == "MERGE" and (e["h"] != e["h0"] or e["t"] != e["t0"]):
                    false_vals.update([x for x in (e["h"], e["t"]) if x not in (e["h0"], e["t0"])])
            for P in provs:
                ctx = kg.serialize(idx, ranked, P)
                rows.append(dict(qid=q["id"], qtype=q["type"], op=op, rate=rate, seed=seed, retriever=R, prov=P,
                                 n_ctx=len(ranked), n_seeds=len(seeds), sup_frac=frac, sup_all=allsup, touched=touched,
                                 false_vals=sorted(false_vals), prompt=qa.gen_prompt(ctx, q["question"]),
                                 question=q["question"], gold=q["answer"]))
    return rows

def generate(rows, model, workers=8, max_tokens=600):
    res = llm.batch(model, [r["prompt"] for r in rows], system=qa.SYS_GEN, max_tokens=max_tokens, workers=workers)
    out = []
    for r, x in zip(rows, res):
        a = qa.parse_answer(x["text"])
        out.append(dict({k: v for k, v in r.items() if k != "prompt"}, model=model, raw=x["text"][-400:], answer=a,
                        abstain=qa.is_abstain(a), em=qa.em(a, r["gold"]), f1=qa.f1(a, r["gold"]),
                        tok_in=x["in"], tok_out=x["out"], err=x.get("error", ""),
                        phash=hashlib.md5(r["prompt"].encode()).hexdigest()))
    return out

def adopted(answer, false_vals, correct):
    if correct or not false_vals: return False
    a = qa._norm(answer)
    return any(a and (a == qa._norm(v) or (len(a) > 3 and (a in qa._norm(v) or qa._norm(v) in a))) for v in false_vals)
