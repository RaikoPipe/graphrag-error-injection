"""Statistical analysis for the error-injection study. Writes results/analysis/*.csv and summary.json."""
import os, sys, glob, json, itertools, warnings, math
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
import runner
OUT = "results/analysis"; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
OMI = {"ENT_DEL", "EDGE_DEL", "SPLIT"}; COM = {"MERGE", "REL_CORRUPT", "SPURIOUS"}
MNAME = {"claude-haiku-4-5-20251001": "Claude Haiku 4.5", "claude-sonnet-4-6": "Claude Sonnet 4.6",
         "ollama:gpt-oss:120b": "gpt-oss-120b", "ollama:gpt-oss:20b": "gpt-oss-20b", "ollama:gemma4:31b": "Gemma 4 31B"}

# ---------- load ----------
J = pd.concat([pd.read_parquet(f) for f in glob.glob("results/judgments_*deepseek*.parquet")]).drop_duplicates(["question", "gold", "answer"], keep="last")
G = pd.concat([pd.read_parquet(f) for f in glob.glob("results/gen_*.parquet")], ignore_index=True)
G = G[G.err == ""].copy()
G = G.merge(J, on=["question", "gold", "answer"], how="left")
G["correct"] = G["correct"].astype(float)
G["model_name"] = G.model.map(MNAME)
pres = pd.read_parquet("results/presence.parquet")
G = G.merge(pres[["qid", "op", "rate", "retriever", "present_all", "linked"]], on=["qid", "op", "rate", "retriever"], how="left")
S0 = {"n_presence_missing": int(G[~G.op.isin(["DENSE", "CLOSED", "NATURAL"])].present_all.isna().sum())}
G["present_all"] = G.present_all.fillna(False).astype(bool)
G["adopted"] = [runner.adopted(a, list(fv) if fv is not None else [], bool(c)) for a, fv, c in zip(G.answer, G.false_vals, G.correct.fillna(0))]
G["outcome"] = np.where(G.correct == 1, "correct", np.where(G.abstain, "abstain", np.where(G.adopted, "adopted", "wrong")))
G["opclass"] = np.where(G.op.isin(OMI), "omission", np.where(G.op.isin(COM), "commission", G.op))
clean_ans = set(pres[(pres.op == "CLEAN") & (pres.retriever == "KHOP") & pres.present_all].qid)
G["clean_answerable"] = G.qid.isin(clean_ans)
G.to_parquet(f"{OUT}/all_rows.parquet")
S = {"n_rows": int(len(G)), "n_unjudged": int(G.correct.isna().sum()), **S0}
G = G[G.correct.notna()]
grid = G[~G.op.isin(["DENSE", "CLOSED", "NATURAL"])]

def boot_mean(df, col="correct", B=2000):
    """question-cluster bootstrap CI of a mean."""
    g = df.groupby("qid")[col].agg(["sum", "count"]); s, c = g["sum"].values, g["count"].values
    idx = rng.integers(0, len(g), (B, len(g)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return df[col].mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

# ---------- T1: accuracy table ----------
rows = []
for keys, d in grid.groupby(["model_name", "retriever", "prov", "op", "rate"]):
    m, lo, hi = boot_mean(d)
    rows.append(dict(zip(["model", "retriever", "prov", "op", "rate"], keys), acc=m, lo=lo, hi=hi, n=len(d),
                     abstain=d.abstain.mean(), adopted=d.adopted.mean(), em=d.em.mean(), support=d.present_all.mean()))
T1 = pd.DataFrame(rows); T1.to_csv(f"{OUT}/T1_accuracy.csv", index=False)
base = []
BB = G[G.op.isin(["DENSE", "CLOSED", "NATURAL"])].copy(); BB["extractor"] = BB["extractor"].fillna("-") if "extractor" in BB else "-"
for keys, d in BB.groupby(["model_name", "op", "prov", "extractor"]):
    m, lo, hi = boot_mean(d); base.append(dict(zip(["model", "op", "prov", "extractor"], keys), acc=m, lo=lo, hi=hi, n=len(d), abstain=d.abstain.mean()))
pd.DataFrame(base).to_csv(f"{OUT}/T_baselines.csv", index=False)

# ---------- H1: dose-response slopes (GEE logit, exchangeable within question) ----------
def gee(formula, d, fam=sm.families.Binomial()):
    import re as _re
    cols = [c for c in set(_re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula)) if c in d.columns]
    d = d.dropna(subset=cols + ["qid"]).reset_index(drop=True)
    for c in cols:
        if d[c].dtype == bool: d[c] = d[c].astype(int)
    return smf.gee(formula, "qid", d, family=fam, cov_struct=sm.cov_struct.Exchangeable()).fit()
H = grid[grid.model == "claude-haiku-4-5-20251001"].copy()
for op in ["ENT_DEL", "EDGE_DEL", "SPLIT", "MERGE", "REL_CORRUPT", "SPURIOUS"]:
    H[f"r_{op}"] = np.where(H.op == op, H.rate, 0.0)
rterms = " + ".join(f"r_{o}" for o in ["ENT_DEL", "EDGE_DEL", "SPLIT", "MERGE", "REL_CORRUPT", "SPURIOUS"])
m1 = gee(f"correct ~ {rterms} + C(retriever) + prov + C(qtype)", H)
slopes = pd.DataFrame({"coef": m1.params, "se": m1.bse, "p": m1.pvalues}).loc[[f"r_{o}" for o in ["ENT_DEL", "EDGE_DEL", "SPLIT", "MERGE", "REL_CORRUPT", "SPURIOUS"]]]
from statsmodels.stats.multitest import multipletests
slopes["p_holm"] = multipletests(slopes.p, method="holm")[1]
slopes["OR_per_0.1"] = np.exp(slopes.coef * 0.1)
slopes.to_csv(f"{OUT}/H1_slopes_accuracy.csv")
S["H1_model_terms"] = {k: float(v) for k, v in m1.params.items()}
# retrieval-level slopes on gold-support (3 seeds, deterministic grid)
R = pd.read_parquet("results/retrieval_grid.parquet")
rs = []
for ret in ["KHOP", "PPR"]:
    for op in ["ENT_DEL", "EDGE_DEL", "SPLIT", "MERGE", "REL_CORRUPT", "SPURIOUS"]:
        d = R[(R.retriever == ret) & (R.op.isin([op, "CLEAN"]))].copy(); d["sup"] = d.sup_all.astype(float)
        f = smf.ols("sup ~ rate", d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.qid)[0]})
        rs.append(dict(retriever=ret, op=op, slope=f.params["rate"], se=f.bse["rate"], sup_at_0_3=d[d.rate == 0.3].sup.mean()))
pd.DataFrame(rs).to_csv(f"{OUT}/H1_slopes_support.csv", index=False)

# ---------- H2: variance decomposition (Shapley R^2 over factors, condition-level, rate<=0.3) ----------
def shapley_r2(df, y, factors):
    def r2(fs):
        if not fs: return 0.0
        X = pd.get_dummies(df[list(fs)].astype(str), drop_first=True).astype(float)
        return sm.OLS(df[y], sm.add_constant(X)).fit().rsquared
    out = {}
    for f in factors:
        others = [x for x in factors if x != f]; val = 0.0
        for k in range(len(others) + 1):
            for sub in itertools.combinations(others, k):
                w = math.factorial(k) * math.factorial(len(factors) - k - 1) / math.factorial(len(factors))
                val += w * (r2(list(sub) + [f]) - r2(list(sub)))
        out[f] = val
    out["total_R2"] = r2(factors); return out
T1["graph"] = T1.op + "@" + T1.rate.astype(str)
a = T1[(T1.model == "Claude Haiku 4.5") & (T1.rate <= 0.3)]
b = T1[(T1.retriever == "KHOP") & (T1.rate.isin([0.0, 0.1, 0.3])) & (T1.model != "Claude Sonnet 4.6")]
c = T1[(T1.retriever == "KHOP") & (T1.rate.isin([0.0, 0.3]))]
S["H2_shapley_haiku"] = shapley_r2(a, "acc", ["graph", "retriever", "prov"])
S["H2_shapley_generators"] = shapley_r2(b, "acc", ["graph", "model", "prov"])
S["H2_shapley_all5"] = shapley_r2(c, "acc", ["graph", "model", "prov"])
# question-level complement: share of deviance explained (McFadden) by each block, Haiku
def mcf(formula, d):
    f = smf.logit(formula, d).fit(disp=0); return f.prsquared
hq = H[H.rate <= 0.3].assign(gcond=lambda x: x.op + "@" + x.rate.astype(str))
S["H2_mcfadden_haiku"] = {"graph": mcf("correct ~ C(gcond)", hq), "retriever": mcf("correct ~ C(retriever)", hq),
                          "prov": mcf("correct ~ prov", hq), "qtype": mcf("correct ~ C(qtype)", hq)}

# ---------- H3 + H6: mediation decomposition ----------
def _med(s0, c0, s1, c1):
    p0, p1 = s0.mean(-1), s1.mean(-1)
    n0s = s0.sum(-1); a0s = (c0 * s0).sum(-1) / np.maximum(n0s, 1)
    n0n = (1 - s0).sum(-1); a0n = (c0 * (1 - s0)).sum(-1) / np.maximum(n0n, 1)
    drop = c0.mean(-1) - c1.mean(-1); ret = (p0 - p1) * (a0s - a0n)
    return drop, ret, drop - ret
med = []
for (m, ret, prov), dm in grid.groupby(["model_name", "retriever", "prov"]):
    d0 = dm[dm.op == "CLEAN"].drop_duplicates("qid").set_index("qid")
    for (op, rate), d1 in dm[dm.op != "CLEAN"].groupby(["op", "rate"]):
        d1 = d1.drop_duplicates("qid").set_index("qid"); q = d0.index.intersection(d1.index)
        s0, c0 = d0.loc[q, "present_all"].values.astype(float), d0.loc[q, "correct"].values
        s1, c1 = d1.loc[q, "present_all"].values.astype(float), d1.loc[q, "correct"].values
        drop, rt, gn = _med(s0, c0, s1, c1)
        ix = rng.integers(0, len(q), (1000, len(q)))
        bd, br, bg = _med(s0[ix], c0[ix], s1[ix], c1[ix])
        sh = np.where(np.abs(bd) > 1e-9, br / np.where(np.abs(bd) > 1e-9, bd, 1), np.nan)
        med.append(dict(model=m, retriever=ret, prov=prov, op=op, rate=rate, drop=drop, retrieval=rt, generation=gn,
                        drop_lo=np.percentile(bd, 2.5), drop_hi=np.percentile(bd, 97.5),
                        ret_share=rt / drop if abs(drop) > 1e-9 else np.nan, ret_share_lo=np.nanpercentile(sh, 2.5), ret_share_hi=np.nanpercentile(sh, 97.5)))
MED = pd.DataFrame(med); MED.to_csv(f"{OUT}/H3_mediation.csv", index=False)

# ---------- H4: outcome distribution by operator class ----------
h4 = H[(H.rate.isin([0.1, 0.2, 0.3, 0.5])) & (H.prov == False)]
ct = pd.crosstab(h4.opclass, h4.outcome); chi = stats.chi2_contingency(ct)
S["H4_chi2"] = dict(chi2=float(chi[0]), dof=int(chi[2]), p=float(chi[1]))
pd.crosstab([h4.op, h4.rate], h4.outcome, normalize="index").to_csv(f"{OUT}/H4_outcomes_by_op.csv")
ct.to_csv(f"{OUT}/H4_outcome_counts.csv")
# among wrong answers (not correct): share abstain vs adopted per class
w = h4[h4.correct == 0]
S["H4_wrong_share"] = w.groupby("opclass").outcome.value_counts(normalize=True).unstack().to_dict(orient="index")

# ---------- H5: provenance moderation ----------
H5d = H[H.op != "CLEAN"].copy(); H5d["commission"] = H5d.op.isin(COM).astype(int); H5d["prov_i"] = H5d.prov.astype(int)
m5 = gee("correct ~ rate * commission * prov_i + C(retriever) + C(qtype)", H5d)
S["H5_terms"] = {k: dict(coef=float(m5.params[k]), p=float(m5.pvalues[k])) for k in m5.params.index if "prov" in k}
m5b = gee("adopted ~ prov_i + C(op) + rate + C(retriever)", H[H.op.isin(["REL_CORRUPT", "MERGE", "SPURIOUS"])].assign(prov_i=lambda x: x.prov.astype(int)))
S["H5_adoption_prov"] = dict(coef=float(m5b.params["prov_i"]), p=float(m5b.pvalues["prov_i"]), OR=float(np.exp(m5b.params["prov_i"])))
pe = []
for (m, op, rate, ret), d in grid[grid.op != "CLEAN"].groupby(["model_name", "op", "rate", "retriever"]):
    a0 = d[~d.prov]; a1 = d[d.prov]
    if len(a0) and len(a1):
        pe.append(dict(model=m, op=op, rate=rate, retriever=ret, acc_noprov=a0.correct.mean(), acc_prov=a1.correct.mean(),
                       delta=a1.correct.mean() - a0.correct.mean(), adopt_noprov=a0.adopted.mean(), adopt_prov=a1.adopted.mean()))
pd.DataFrame(pe).to_csv(f"{OUT}/H5_provenance_effects.csv", index=False)

# ---------- H6: generator moderation ----------
g6 = grid[(grid.retriever == "KHOP") & (grid.rate.isin([0.0, 0.3]))].copy()
g6["cond"] = np.where(g6.op == "CLEAN", "clean", np.where(g6.op.isin(OMI), "omission", "commission"))
S["H6_acc_by_model_cond"] = g6.groupby(["model_name", "cond"]).correct.mean().unstack().to_dict(orient="index")
g6b = g6[g6.op != "CLEAN"].copy(); g6b["commission"] = g6b.op.isin(COM).astype(int)
m6 = gee("correct ~ C(model_name) * commission + prov + C(qtype)", g6b)
S["H6_interaction_p"] = {k: float(v) for k, v in m6.pvalues.items() if ":" in k}
S["H6_med_by_model"] = MED[(MED.retriever == "KHOP") & (MED.rate == 0.3)].groupby(["model", "op"])[["drop", "retrieval", "generation"]].mean().reset_index().to_dict(orient="records")

# ---------- sensitivity: clean-answerable subset ----------
sens = grid[grid.clean_answerable].groupby(["model_name", "retriever", "prov", "op", "rate"]).correct.mean().reset_index()
sens.to_csv(f"{OUT}/sens_clean_answerable.csv", index=False)
S["n_clean_answerable"] = len(clean_ans)
json.dump(S, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
print("DONE", S["n_rows"], S["n_unjudged"])

# ---------- retrieval-level support table ----------
R["sup"] = R.sup_all.astype(float)
TS = R.groupby(["retriever", "op", "rate"]).agg(support=("sup", "mean"), support_frac=("sup_frac", "mean"), touched=("touched", "mean")).reset_index()
TS["sd_seeds"] = R.groupby(["retriever", "op", "rate", "seed"]).sup.mean().groupby(["retriever", "op", "rate"]).std().values
TS.to_csv(f"{OUT}/T_support.csv", index=False)

# ---------- H7: natural graphs ----------
import kg
Q = json.load(open("questions.json")); paras = json.load(open("paragraphs.json"))
P = pd.read_csv("results/extractor_profiles.csv", index_col=0)
trips = {"gpt-oss-120b": json.load(open("extract_gptoss120b_all.json")), "claude-haiku-4.5": json.load(open("extract_haiku45_all.json"))}
res = kg.Resolver(list(paras))
def covq(trip):
    pairs = set()
    for ti, tr in trip.items():
        for _, h, r, t in tr:
            a, b = res(h), res(t)
            if a and b: pairs.add(frozenset([kg.norm(a), kg.norm(b)]))
    rows = []
    for q in Q:
        c = [frozenset([kg.norm(res(h) or h), kg.norm(res(t) or t)]) in pairs for h, r, t in q["evidences"]]
        rows.append(dict(qid=q["id"], frac=np.mean(c), all=all(c), k=len(c)))
    return pd.DataFrame(rows)
HK = G[(G.model == "claude-haiku-4-5-20251001") & (G.retriever == "KHOP") & ~G.op.isin(["DENSE", "CLOSED", "NATURAL"])].copy()
for o in ["ENT_DEL", "EDGE_DEL", "SPLIT", "MERGE", "REL_CORRUPT", "SPURIOUS"]: HK[f"r_{o}"] = np.where(HK.op == o, HK.rate, 0.0)
NAT = G[G.op == "NATURAL"]
h7 = []
for prov in [False, True]:
    hk = HK[HK.prov == prov]
    m = smf.logit("correct ~ " + " + ".join(f"r_{o}" for o in ["ENT_DEL", "EDGE_DEL", "SPLIT", "MERGE", "REL_CORRUPT", "SPURIOUS"]), hk).fit(disp=0)
    a_s = hk[hk.present_all].correct.mean(); a_n = hk[~hk.present_all].correct.mean()
    for ex in trips:
        pr = P.loc[ex]
        x = pd.DataFrame([dict(r_ENT_DEL=0, r_EDGE_DEL=1 - pr.coverage_lenient, r_SPLIT=pr.split_rate, r_MERGE=0, r_REL_CORRUPT=pr.tail_conflict, r_SPURIOUS=pr.unsupported)])
        sf = m.get_prediction(x).summary_frame(alpha=0.05)
        cq = covq(trips[ex]); p_all = cq["all"].mean(); p_ind = np.mean([cq.frac.mean() ** k for k in cq.k])
        obs = NAT[(NAT.extractor == ex) & (NAT.prov == prov)]
        om, olo, ohi = boot_mean(obs)
        h7.append(dict(prov=prov, extractor=ex, pred_independent=float(sf.iloc[0, 0]), pred_lo=float(sf.iloc[0, 2]), pred_hi=float(sf.iloc[0, 3]),
                       pred_mediation=p_all * a_s + (1 - p_all) * a_n, q_all_covered=p_all, q_all_covered_indep=p_ind, fact_cov=cq.frac.mean(),
                       observed=om, obs_lo=olo, obs_hi=ohi, acc_given_support=a_s, acc_given_nosupport=a_n))
pd.DataFrame(h7).to_csv(f"{OUT}/H7_natural.csv", index=False)
print("H7 done")
