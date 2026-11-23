import json, numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
A = "results/analysis/"; F = "../paper/figures/"; os.makedirs(F, exist_ok=True)
T = pd.read_csv(A + "T1_accuracy.csv"); TS = pd.read_csv(A + "T_support.csv"); M = pd.read_csv(A + "H3_mediation.csv")
O = pd.read_csv(A + "H4_outcomes_by_op.csv"); B = pd.read_csv(A + "T_baselines.csv"); N = pd.read_csv(A + "H7_natural.csv")
S = json.load(open(A + "summary.json")); P = pd.read_csv("results/extractor_profiles.csv", index_col=0)
OPS = ["ENT_DEL", "EDGE_DEL", "SPLIT", "MERGE", "REL_CORRUPT", "SPURIOUS"]
LBL = {"ENT_DEL": "Entity deletion", "EDGE_DEL": "Edge deletion", "SPLIT": "Entity split", "MERGE": "Entity merge",
       "REL_CORRUPT": "Relation corruption", "SPURIOUS": "Spurious edges"}
FOCAL = "#1f4e79"; C2 = "#7fa7cf"; C3 = "#b5651d"; C4 = "#e3b58a"; GREY = "#8c8c8c"
def apply_figure_style(sizes=(8, 7, 6)):
    base, mid, small = sizes
    mpl.rcParams.update({"font.size": base, "axes.titlesize": base, "axes.labelsize": base, "legend.fontsize": mid,
                         "xtick.labelsize": small, "ytick.labelsize": small, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight", "pdf.fonttype": 42})
apply_figure_style(sizes=(8, 7, 6))
HK = "Claude Haiku 4.5"
rates = [0.0, 0.1, 0.2, 0.3, 0.5]
def series(ret, prov, op, model=HK):
    d = T[(T.model == model) & (T.retriever == ret) & (T.prov == prov) & (T.op.isin([op, "CLEAN"]))].sort_values("rate")
    return d.rate.values, d.acc.values, d.lo.values, d.hi.values

# ---------- Fig 2: dose-response ----------
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.3), sharex=True, sharey=True)
dense = B[(B.model == HK) & (B.op == "DENSE")].acc.iloc[0]; closed = B[(B.model == HK) & (B.op == "CLOSED")].acc.iloc[0]
for ax, op in zip(axes.ravel(), OPS):
    for ret, prov, c, ls, lab in [("KHOP", False, FOCAL, "-", "k-hop, triples"), ("KHOP", True, C2, "-", "k-hop, triples + source"),
                                  ("PPR", False, C3, "--", "PPR, triples"), ("PPR", True, C4, "--", "PPR, triples + source")]:
        x, y, lo, hi = series(ret, prov, op)
        ax.fill_between(x, lo, hi, color=c, alpha=0.15, lw=0); ax.plot(x, y, ls, color=c, marker="o", ms=2.5, lw=1.4 if c == FOCAL else 1.0, label=lab)
    s = TS[(TS.retriever == "KHOP") & (TS.op.isin([op, "CLEAN"])) & (TS.rate.isin(rates))].sort_values("rate")
    ax.plot(s.rate, s.support, ":", color=GREY, lw=1.0, label="gold evidence retrieved (k-hop)")
    ax.axhline(dense, color="#555555", lw=0.6, ls=(0, (1, 2))); ax.axhline(closed, color="#555555", lw=0.6, ls=(0, (4, 2)))
    ax.set_title(f"$\\bf{{{'abcdef'[OPS.index(op)]}}}$  " + LBL[op], loc="left"); ax.set_ylim(0, 1); ax.set_xticks(rates); ax.margins(x=0.04)
for ax in axes[1]: ax.set_xlabel("Error rate p")
for ax in axes[:, 0]: ax.set_ylabel("Accuracy / support")
axes[1, 2].text(0.52, dense + 0.02, "dense passage RAG", fontsize=6, color="#555555", ha="right")
axes[1, 2].text(0.52, closed + 0.02, "closed-book", fontsize=6, color="#555555", ha="right")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(bottom=0.17, wspace=0.06, hspace=0.3)
fig.savefig(F + "fig2_dose_response.pdf"); fig.savefig(F + "fig2_dose_response.png", dpi=300)

# ---------- Fig 3: mediation + outcomes at p=0.3 ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1, 1.15]})
m = M[(M.model == HK) & (M.retriever == "KHOP") & (M.prov == False) & (M.rate == 0.3)].set_index("op").loc[OPS]
y = np.arange(len(OPS))
a1.barh(y, m.retrieval, color=FOCAL, label="retrieval loss (gold evidence not retrieved)")
a1.barh(y, m.generation, left=m.retrieval.clip(lower=0), color="#8e6bb0", label="generation loss (evidence retrieved, answer wrong)")
a1.errorbar(m["drop"], y, xerr=[m["drop"] - m.drop_lo, m.drop_hi - m["drop"]], fmt="none", ecolor="black", lw=0.7, capsize=1.5)
a1.set_yticks(y, [LBL[o] for o in OPS]); a1.invert_yaxis(); a1.set_xlabel("Accuracy drop vs. clean graph (p = 0.3)")
a1.set_title("$\\bf{a}$  Omission errors act through retrieval", loc="left"); a1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.22), fontsize=6); a1.set_xlim(-0.02, 0.69); a1.set_xticks([0, 0.2, 0.4, 0.6])
a1.axvline(0, color="black", lw=0.5); a1.margins(x=0.04)
o = O[O.rate == 0.3].set_index("op").loc[OPS]
cols = [("correct", "#4d7f4d", "correct"), ("abstain", "#bdbdbd", "abstained"), ("adopted", "#c0392b", "adopted injected false value"), ("wrong", "#555555", "other wrong")]
left = np.zeros(len(OPS))
for k, c, lab in cols:
    a2.barh(y, o[k], left=left, color=c, label=lab); left += o[k].values
a2.set_yticks(y, [""] * len(OPS)); a2.invert_yaxis(); a2.set_xlim(0, 1); a2.set_xlabel("Share of answers (p = 0.3, triples only)")
a2.set_title("$\\bf{b}$  Commission errors produce confident wrong answers", loc="left")
a2.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=6)
fig.subplots_adjust(wspace=0.08, bottom=0.33)
fig.savefig(F + "fig3_mechanism.pdf"); fig.savefig(F + "fig3_mechanism.png", dpi=300)

# ---------- Fig 4: variance decomposition + generators ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [0.8, 1.6]})
sh = [("Graph errors\n(operator × rate)", S["H2_shapley_haiku"]["graph"], S["H2_shapley_all5"]["graph"]),
      ("Retriever", S["H2_shapley_haiku"]["retriever"], np.nan), ("Provenance", S["H2_shapley_haiku"]["prov"], S["H2_shapley_all5"]["prov"]),
      ("Generator", np.nan, S["H2_shapley_all5"]["model"])]
yy = np.arange(len(sh)); w = 0.38
a1.barh(yy - w / 2, [s[1] for s in sh], w, color=FOCAL, label="Haiku grid (p ≤ 0.3)")
a1.barh(yy + w / 2, [s[2] for s in sh], w, color=C3, label="5 generators, k-hop (p ∈ {0, 0.3})")
for i, s in enumerate(sh):
    for off, v in [(-w / 2, s[1]), (w / 2, s[2])]:
        if np.isnan(v): a1.text(0.01, i + off, "not varied", va="center", fontsize=5.5, color=GREY)
        else: a1.text(v + 0.01, i + off, f"{v:.2f}", va="center", fontsize=6)
a1.set_yticks(yy, [s[0] for s in sh]); a1.invert_yaxis(); a1.set_xlim(0, 1.1); a1.set_xlabel("Shapley share of R²")
a1.set_title("$\\bf{a}$  Graph quality dominates variance", loc="left"); a1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.35, -0.25), fontsize=5.5)
models = ["Claude Haiku 4.5", "Claude Sonnet 4.6", "gpt-oss-120b", "Gemma 4 31B", "gpt-oss-20b"]
mc = {"Claude Haiku 4.5": FOCAL, "Claude Sonnet 4.6": C2, "gpt-oss-120b": C3, "Gemma 4 31B": "#4d7f4d", "gpt-oss-20b": C4}
conds = ["CLEAN"] + OPS
for j, md in enumerate(models):
    vals = []
    for c in conds:
        d = T[(T.model == md) & (T.retriever == "KHOP") & (T.prov == False) & (T.op == c) & (T.rate == (0.0 if c == "CLEAN" else 0.3))]
        vals.append(d.acc.iloc[0] if len(d) else np.nan)
    a2.plot(np.arange(len(conds)) + (j - 2) * 0.09, vals, "o", ms=3.2, color=mc[md], label=md)
a2.set_xticks(np.arange(len(conds)), ["Clean"] + [LBL[o].replace(" ", "\n", 1) for o in OPS]); a2.set_ylim(0.2, 1.0)
a2.set_ylabel("Accuracy"); a2.set_title("$\\bf{b}$  Generators barely differ under graph errors (p = 0.3)", loc="left")
a2.legend(frameon=False, ncol=5, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.25)); a2.margins(x=0.03)
fig.subplots_adjust(wspace=0.55, bottom=0.33)
fig.savefig(F + "fig4_attribution.pdf"); fig.savefig(F + "fig4_attribution.png", dpi=300)

# ---------- Fig 5: natural extractors ----------
fig, ax = plt.subplots(figsize=(3.4, 2.4))
lab = {"gpt-oss-120b": "gpt-oss-120b graph", "claude-haiku-4.5": "Claude Haiku 4.5 graph"}
yy = 0
ticks = []
for ex in ["gpt-oss-120b", "claude-haiku-4.5"]:
    for prov in [False, True]:
        r = N[(N.extractor == ex) & (N.prov == prov)].iloc[0]
        ax.errorbar(r.observed, yy, xerr=[[r.observed - r.obs_lo], [r.obs_hi - r.observed]], fmt="o", color=FOCAL, ms=4, lw=0.8, capsize=1.5, label="observed" if yy == 0 else None)
        ax.plot(r.pred_independent, yy, "x", color=C3, ms=5, label="predicted: independent errors" if yy == 0 else None)
        ax.plot(r.pred_mediation, yy, "D", color="#4d7f4d", ms=3.5, mfc="none", label="predicted: measured per-question coverage" if yy == 0 else None)
        ticks.append(f"{lab[ex]}\n{'triples + source' if prov else 'triples only'}"); yy += 1
ax.set_yticks(range(yy), ticks); ax.invert_yaxis(); ax.set_xlabel("Accuracy (Claude Haiku 4.5, k-hop)"); ax.set_xlim(0.2, 0.8)
ax.legend(frameon=False, fontsize=5.5, loc="lower right"); ax.set_title("Injected curves bound natural-graph accuracy from below", loc="left")
fig.savefig(F + "fig5_natural.pdf", bbox_inches="tight"); fig.savefig(F + "fig5_natural.png", dpi=300, bbox_inches="tight")
print("ok")
