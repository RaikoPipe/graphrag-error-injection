"""Reproduces the study steps that were originally run interactively.

Usage: python run_extras.py <step> [<step> ...]
Steps: prepare | extract | profiles | natural | baselines | judge_validation
All LLM calls go through llm.py and are served from cache/llm_cache.sqlite when present.
"""
import os, sys, json, re, random, collections
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
os.environ.setdefault("LLM_CACHE", "cache/llm_cache.sqlite")
import numpy as np, pandas as pd
from rapidfuzz import fuzz
import llm, kg, qa, runner, extract

HF_VAL = "https://huggingface.co/datasets/framolfese/2WikiMultihopQA/resolve/main/data/validation-00000-of-00001.parquet"

def load():
    return json.load(open("questions.json")), json.load(open("paragraphs.json"))

# ---------------------------------------------------------------- prepare
def prepare():
    """Stratified sample of 600 dev questions (150 per type, seed 42) and the shared paragraph pool."""
    import requests
    os.makedirs("data", exist_ok=True)
    p = "data/2wiki_val.parquet"
    if not os.path.exists(p):
        open(p, "wb").write(requests.get(HF_VAL, timeout=300).content)
    df = pd.read_parquet(p)
    samp = pd.concat([g.sample(150, random_state=42) for _, g in df.groupby("type")]).reset_index(drop=True)
    paras, Q = {}, []
    for _, r in samp.iterrows():
        titles = list(r.context["title"]); sents = [list(s) for s in r.context["sentences"]]
        for t, s in zip(titles, sents):
            paras.setdefault(t, s)
        Q.append(dict(id=r.id, question=r.question, answer=r.answer, type=r.type,
                      evidences=[list(map(str, e)) for e in r.evidences],
                      supporting=list(zip(list(r.supporting_facts["title"]), [int(x) for x in r.supporting_facts["sent_id"]])),
                      titles=titles))
    json.dump(Q, open("questions.json", "w")); json.dump(paras, open("paragraphs.json", "w"))
    print("prepare:", len(Q), "questions,", len(paras), "paragraphs")

def supporting_titles(Q, paras):
    return sorted({t for q in Q for t, _ in q["supporting"] if t in paras})

# ---------------------------------------------------------------- extract
def do_extract():
    Q, paras = load()
    for model, sub, fn in [("ollama:gpt-oss:120b", paras, "extract_gptoss120b_all.json"),
                           ("claude-haiku-4-5-20251001", paras, "extract_haiku45_all.json"),
                           ("ollama:gemma4:31b", {t: paras[t] for t in supporting_titles(Q, paras)}, "extract_gemma4_sup.json")]:
        trip, errs, tok = extract.extract_all(model, sub, workers=8)
        json.dump(trip, open(fn, "w")); print("extract:", model, sum(len(v) for v in trip.values()), "triples, errors", errs)

# ---------------------------------------------------------------- profiles
VT = """Paragraph (title: {ti}):
{p}

Extracted triple: ({h}; {r}; {t})

Is this triple explicitly stated by or directly entailed by the paragraph (with pronouns resolved)? Wrong subject, wrong object, or a fact not in the text counts as NOT supported.
Reply with exactly one word: SUPPORTED or UNSUPPORTED."""

def profiles():
    """Natural error profile of each extractor on the supporting paragraphs (Table 'natural' in the paper)."""
    Q, paras = load()
    sup = supporting_titles(Q, paras)
    _tr = kg.Resolver(list(paras))
    titleres = lambda s: _tr(s) or s
    gold = list({(h, r, t) for q in Q for h, r, t in q["evidences"]})
    def profile(trip):
        T = [(h, r, t) for ti in sup for (_, h, r, t) in trip.get(ti, [])]
        by_hr = collections.defaultdict(set); pairs = set(); pairs_any = set()
        for h, r, t in T:
            nh, nt = kg.norm(titleres(h)), kg.norm(titleres(t))
            by_hr[(nh, r)].add(nt); pairs.add((nh, r, nt)); pairs_any.update([(nh, nt), (nt, nh)])
        cs = np.mean([(kg.norm(titleres(h)), r, kg.norm(titleres(t))) in pairs for h, r, t in gold])
        cl = np.mean([(kg.norm(titleres(h)), kg.norm(titleres(t))) in pairs_any for h, r, t in gold])
        conf = []
        for h, r, t in gold:
            k = (kg.norm(titleres(h)), r)
            if k in by_hr:
                conf.append(kg.norm(titleres(t)) not in by_hr[k] and not any(fuzz.token_set_ratio(kg.norm(t), x) >= 90 for x in by_hr[k]))
        return dict(n_triples=len(T), coverage_strict=cs, coverage_lenient=cl, tail_conflict=np.mean(conf), n_conf_base=len(conf))
    def is_variant(k, ng):
        a, b = set(k.split()), set(ng.split())
        if k == ng: return False
        sub_ = (a < b or b < a) and abs(len(a) - len(b)) == 1 and max(len(a), len(b)) <= 4 and min(len(a), len(b)) >= 2
        return sub_ or (fuzz.ratio(k, ng) >= 90 and len(ng) >= 8)
    def split_rate(trip):
        surf = set()
        for ti in sup:
            for _, h, r, t in trip.get(ti, []): surf.update([kg.norm(titleres(h)), kg.norm(titleres(t))])
        bytok = collections.defaultdict(set)
        for k in surf:
            for w in k.split(): bytok[w].add(k)
        ents = sorted({x for g in gold for x in (g[0], g[2]) if not re.match(r"^\d", x)})
        out = []
        for g in ents:
            ng = kg.norm(titleres(g))
            if ng not in surf: continue
            cands = set().union(*[bytok[w] for w in ng.split()]) if ng.split() else set()
            out.append(any(is_variant(k, ng) for k in cands))
        return np.mean(out), len(out)
    trips = [("gpt-oss-120b", json.load(open("extract_gptoss120b_all.json"))),
             ("claude-haiku-4.5", json.load(open("extract_haiku45_all.json"))),
             ("gemma4-31b", json.load(open("extract_gemma4_sup.json")))]
    prof = {}
    for m, t in trips:
        prof[m] = profile(t); prof[m]["split_rate"], prof[m]["n_ent"] = split_rate(t)
    rng = random.Random(7)
    for m, t in trips:
        pool = [(ti, x) for ti in sup for x in t.get(ti, [])]
        smp = rng.sample(pool, 200)
        res = llm.batch("claude-sonnet-4-6", [VT.format(ti=ti, p=" ".join(paras[ti]), h=x[1], r=x[2], t=x[3]) for ti, x in smp], max_tokens=10, workers=8)
        uns = [("UNSUPPORTED" in r["text"].upper()) for r in res]
        prof[m]["unsupported"] = np.mean(uns); prof[m]["unsupported_se"] = np.std(uns) / np.sqrt(len(uns))
    P = pd.DataFrame(prof).T; P.to_csv("results/extractor_profiles.csv"); print(P.round(3).to_string())

# ---------------------------------------------------------------- natural graphs
def build_natural(paras, extracted):
    res = kg.Resolver(list(paras)); E = {}
    for title, trips in extracted.items():
        for sid, h, r, t in trips:
            H, T = res(h), res(t)
            if not H or not T or kg.norm(H) == kg.norm(T) or (H, r, T) in E: continue
            s = paras.get(title, [])
            E[(H, r, T)] = dict(h=H, r=r, t=T, sent=s[sid] if sid < len(s) else "", src=title, gold=[], corrupt=False, spurious=False)
    E = list(E.values())
    for i, e in enumerate(E): e["id"] = i; e["h0"], e["t0"] = e["h"], e["t"]
    return E

def natural():
    """End-to-end runs over graphs built only from extractor output (no gold anchoring)."""
    Q, paras = load(); out = []
    for ename, fn in [("gpt-oss-120b", "extract_gptoss120b_all.json"), ("claude-haiku-4.5", "extract_haiku45_all.json")]:
        ix = kg.Index(build_natural(paras, json.load(open(fn)))); rows = []
        for q in Q:
            rk, seeds = kg.retrieve_khop(ix, q["question"])
            ends = {kg.norm(ix.edges[i]["h"]) for i in rk} | {kg.norm(ix.edges[i]["t"]) for i in rk}
            for P in (False, True):
                rows.append(dict(qid=q["id"], qtype=q["type"], op="NATURAL", extractor=ename, rate=np.nan, seed=0, retriever="KHOP", prov=P,
                                 n_ctx=len(rk), n_seeds=len(seeds), ans_in_ctx=kg.norm(q["answer"]) in ends, false_vals=[],
                                 prompt=qa.gen_prompt(kg.serialize(ix, rk, P), q["question"]), question=q["question"], gold=q["answer"],
                                 sup_frac=np.nan, sup_all=np.nan, touched=np.nan))
        o = runner.generate(rows, "claude-haiku-4-5-20251001", workers=8)
        for x in o: x["extractor"] = ename
        out += o; print("natural:", ename, round(np.mean([x["em"] for x in o]), 3))
    pd.DataFrame(out).to_parquet("results/gen_natural_haiku.parquet")

# ---------------------------------------------------------------- baselines
def baselines():
    """Closed-book and dense passage RAG (TF-IDF top-5) for all five generators."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    Q, paras = load()
    titles = list(paras); docs = [t + ". " + " ".join(paras[t]) for t in titles]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words="english"); X = vec.fit_transform(docs)
    S = (vec.transform([q["question"] for q in Q]) @ X.T).toarray()
    rows = []
    for i, q in enumerate(Q):
        top = np.argsort(-S[i])[:5]; ctx = "\n\n".join(f"[{titles[j]}] " + " ".join(paras[titles[j]]) for j in top)
        sup_t = {t for t, _ in q["supporting"]}; hit = len(sup_t & {titles[j] for j in top}) / max(1, len(sup_t))
        c = dict(qid=q["id"], qtype=q["type"], rate=0.0, seed=0, prov=False, question=q["question"], gold=q["answer"], false_vals=[],
                 sup_frac=hit, sup_all=hit == 1, touched=False, n_ctx=5, n_seeds=0)
        rows.append(dict(c, op="DENSE", retriever="TFIDF", prompt=qa.TMPL_DENSE.format(ctx=ctx, q=q["question"])))
        rows.append(dict(c, op="CLOSED", retriever="NONE", prompt=qa.TMPL_CB.format(q=q["question"])))
    for fn, models in [("results/gen_baselines_anthropic.parquet", ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"]),
                       ("results/gen_baselines_ollama.parquet", ["ollama:gpt-oss:120b", "ollama:gemma4:31b", "ollama:gpt-oss:20b"])]:
        out = []
        for m in models: out += runner.generate(rows, m, workers=6)
        pd.DataFrame(out).to_parquet(fn); print("baselines:", fn, len(out))

# ---------------------------------------------------------------- judge validation
def judge_validation():
    """Re-judges the shipped 300-item validation sample with a second judge and adjudicates disagreements.
    The sample itself (results/judge_validation.csv) was drawn with random_state=1 from the non-trivial
    judgments available at the time of validation."""
    from sklearn.metrics import cohen_kappa_score
    smp = pd.read_csv("results/judge_validation.csv")
    items = list(smp[["question", "gold", "answer"]].itertuples(index=False, name=None))
    smp["sonnet"] = qa.judge("claude-sonnet-4-6", items, workers=8)
    dis = smp[smp.correct != smp.sonnet]
    smp.loc[dis.index, "opus"] = qa.judge("claude-opus-4-5-20251101", list(dis[["question", "gold", "answer"]].itertuples(index=False, name=None)), workers=8)
    smp["ref"] = np.where(smp.correct == smp.sonnet, smp.correct, smp.opus)
    val = dict(n=len(smp), agree=float((smp.correct == smp.sonnet).mean()), kappa=float(cohen_kappa_score(smp.correct, smp.sonnet)),
               n_disagree=len(dis), deepseek_acc_vs_ref=float((smp.correct == smp.ref).mean()), sonnet_acc_vs_ref=float((smp.sonnet == smp.ref).mean()))
    print(val)

STEPS = dict(prepare=prepare, extract=do_extract, profiles=profiles, natural=natural, baselines=baselines, judge_validation=judge_validation)
if __name__ == "__main__":
    for s in sys.argv[1:] or ["profiles"]:
        STEPS[s]()
