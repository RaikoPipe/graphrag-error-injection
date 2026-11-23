"""Gold-anchored graph construction, typed error operators, and graph retrievers."""
import re, json, copy, random, unicodedata, collections
import numpy as np
import scipy.sparse as sp
from rapidfuzz import process, fuzz

# ---------------- normalisation & resolution ----------------
def norm(s):
    s = s.lower().strip()
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

class Resolver:
    """Maps surface strings to canonical names (gold names and paragraph titles take priority)."""
    def __init__(self, canon_names):
        self.by_norm = {}
        for c in canon_names:
            self.by_norm.setdefault(norm(c), c)
        self.keys = list(self.by_norm)
    def __call__(self, s):
        n = norm(s)
        if not n: return None
        if n in self.by_norm: return self.by_norm[n]
        if len(n) >= 6:
            m = process.extractOne(n, self.keys, scorer=fuzz.ratio, score_cutoff=94)
            if m: return self.by_norm[m[0]]
        return s.strip()

def build_graph(Q, paras, extracted):
    """Returns edge list. Each edge: dict(h, r, t, sent, src, gold(list of (qid, k)) )."""
    gold_names = set()
    for q in Q:
        for h, r, t in q["evidences"]:
            gold_names.update([h, t])
    canon = list(gold_names) + list(paras)
    res = Resolver(canon)
    gold_edges, gold_hr, gold_pairs = {}, set(), set()
    for q in Q:
        sup = [(ti, si) for ti, si in q["supporting"] if ti in paras and si < len(paras[ti])]
        for k, (h, r, t) in enumerate(q["evidences"]):
            h, t = res(h), res(t)
            key = (h, r, t)
            if key not in gold_edges:
                # provenance: supporting sentence mentioning the tail (fuzzy), else one from head's paragraph
                cands = [paras[ti][si] for ti, si in sup]
                best = max(cands, key=lambda s: fuzz.partial_ratio(norm(t), norm(s)) + 0.5 * fuzz.partial_ratio(norm(h), norm(s)), default="")
                src = next((ti for ti, si in sup if paras[ti][si] == best), h)
                gold_edges[key] = dict(h=h, r=r, t=t, sent=best, src=src, gold=[])
            gold_edges[key]["gold"].append((q["id"], k))
            gold_hr.add((norm(h), r)); gold_pairs.add((norm(h), norm(t))); gold_pairs.add((norm(t), norm(h)))
    bg, dropped = {}, 0
    for title, trips in extracted.items():
        for sid, h, r, t in trips:
            H, T = res(h), res(t)
            if not H or not T or norm(H) == norm(T): continue
            if (norm(H), norm(T)) in gold_pairs or (norm(H), r) in gold_hr:
                dropped += 1; continue
            key = (H, r, T)
            if key in bg: continue
            sents = paras.get(title, [])
            bg[key] = dict(h=H, r=r, t=T, sent=sents[sid] if sid < len(sents) else "", src=title, gold=[])
    edges = list(gold_edges.values()) + list(bg.values())
    for i, e in enumerate(edges):
        e["id"] = i; e["corrupt"] = False; e["spurious"] = False
        e["h0"], e["t0"] = e["h"], e["t"]
    return edges, dict(n_gold=len(gold_edges), n_bg=len(bg), dropped_dupe=dropped)

def nodes_of(edges):
    return sorted({e["h"] for e in edges} | {e["t"] for e in edges})

# ---------------- error operators ----------------
OPS = ["ENT_DEL", "EDGE_DEL", "SPLIT", "MERGE", "REL_CORRUPT", "SPURIOUS"]
OMISSION = {"ENT_DEL", "EDGE_DEL", "SPLIT"}

def alias(name, rng):
    toks = name.split()
    opts = []
    if len(toks) >= 2:
        opts.append(" ".join([toks[0][0] + "."] + toks[1:]))          # initialism
        opts.append(toks[-1] + ", " + " ".join(toks[:-1]))            # reordered
    opts.append(name + " (entity)")                                  # qualified
    return rng.choice(opts)

_PARTNER_CACHE = {}
def merge_partners(V):
    """Most name-similar other node, searched within a block sharing the first 3 normalised chars (deterministic)."""
    key = len(V)
    if key in _PARTNER_CACHE: return _PARTNER_CACHE[key]
    blocks = collections.defaultdict(list)
    for v in V: blocks[norm(v)[:3]].append(v)
    out = {}
    for b in blocks.values():
        if len(b) < 2: continue
        for v in b:
            m = process.extract(v, b, scorer=fuzz.WRatio, limit=2)
            p = next((c for c, sc, _ in m if c != v), None)
            if p: out[v] = p
    _PARTNER_CACHE[key] = out
    return out

def inject(edges, op, rate, seed):
    """Apply one typed error operator at rate to a deep copy of edges. Returns (edges', log)."""
    rng = random.Random(seed * 1000 + OPS.index(op) * 37 + int(rate * 100))
    E = copy.deepcopy(edges)
    log = dict(op=op, rate=rate, seed=seed, touched_nodes=set(), touched_edges=set())
    if rate == 0: return E, log
    V = nodes_of(E)
    if op == "ENT_DEL":
        kill = set(rng.sample(V, int(rate * len(V))))
        log["touched_nodes"] = kill
        log["touched_edges"] = {e["id"] for e in E if e["h"] in kill or e["t"] in kill}
        E = [e for e in E if e["h"] not in kill and e["t"] not in kill]
    elif op == "EDGE_DEL":
        kill = set(rng.sample([e["id"] for e in E], int(rate * len(E))))
        log["touched_edges"] = kill
        E = [e for e in E if e["id"] not in kill]
    elif op == "SPLIT":
        chosen = rng.sample(V, int(rate * len(V)))
        amap = {v: alias(v, rng) for v in chosen}
        log["touched_nodes"] = set(chosen)
        for e in E:
            for end in ("h", "t"):
                if e[end] in amap and rng.random() < 0.5:
                    e[end] = amap[e[end]]; log["touched_edges"].add(e["id"])
    elif op == "MERGE":
        chosen = rng.sample(V, int(rate * len(V)))
        partners = merge_partners(V)
        # partner = most name-similar other node (confusable entity); survivor name random
        mp = {}
        for v in chosen:
            partner = partners.get(v)
            if partner is None or partner in mp or v in mp: continue
            surv = rng.choice([v, partner])
            mp[v] = surv; mp[partner] = surv
            log["touched_nodes"].update([v, partner])
        for e in E:
            for end in ("h", "t"):
                if e[end] in mp and mp[e[end]] != e[end]:
                    e[end] = mp[e[end]]; log["touched_edges"].add(e["id"])
                elif e[end] in mp:
                    log["touched_edges"].add(e["id"])
        E = [e for e in E if e["h"] != e["t"]]
    elif op == "REL_CORRUPT":
        by_rel = collections.defaultdict(list)
        for e in E: by_rel[e["r"]].append(e["t"])
        for e in rng.sample(E, int(rate * len(E))):
            pool = [t for t in by_rel[e["r"]] if t != e["t"]] or [t for t in rng.sample(V, 5) if t != e["t"]]
            e["t"] = rng.choice(pool); e["corrupt"] = True; log["touched_edges"].add(e["id"])
    elif op == "SPURIOUS":
        n = int(rate * len(E)); nid = max(e["id"] for e in E) + 1
        heads = V
        sent_by_src = collections.defaultdict(list)
        for e in E:
            if e["sent"]: sent_by_src[e["h"]].append(e["sent"])
        new = []
        for i in range(n):
            tmpl = rng.choice(E); h = rng.choice(heads)
            if h == tmpl["t"]: continue
            s = rng.choice(sent_by_src[h]) if sent_by_src[h] else ""
            new.append(dict(id=nid + i, h=h, r=tmpl["r"], t=tmpl["t"], sent=s, src=h, gold=[], corrupt=False,
                            spurious=True, h0=h, t0=tmpl["t"]))
        log["touched_nodes"] = {x["h"] for x in new}
        E = E + new
    return E, log

# ---------------- retrieval ----------------
SYN = {"born": ["date of birth", "place of birth"], "birth": ["date of birth", "place of birth"],
       "die": ["date of death", "place of death", "cause of death"], "died": ["date of death", "place of death", "cause of death"],
       "death": ["date of death", "place of death", "cause of death"], "older": ["date of birth"], "younger": ["date of birth"],
       "earlier": ["date of birth", "publication date"], "later": ["date of birth", "publication date"],
       "first": ["publication date", "date of birth", "inception"], "released": ["publication date"], "come out": ["publication date"],
       "came out": ["publication date"], "directed": ["director"], "director": ["director"], "film": ["director", "publication date"],
       "mother": ["mother"], "father": ["father"], "paternal": ["father"], "maternal": ["mother"], "grandfather": ["father", "mother"],
       "grandmother": ["mother", "father"], "grandchild": ["child"], "child": ["child"], "son": ["child"], "daughter": ["child"],
       "husband": ["spouse"], "wife": ["spouse"], "spouse": ["spouse"], "in-law": ["spouse", "father", "mother"],
       "law": ["spouse", "father", "mother"], "uncle": ["sibling", "father", "mother"], "aunt": ["sibling", "father", "mother"],
       "sibling": ["sibling"], "brother": ["sibling"], "sister": ["sibling"], "country": ["country of citizenship", "country of origin", "country"],
       "nationality": ["country of citizenship"], "same country": ["country of citizenship", "country of origin", "country"],
       "from": ["country of citizenship", "country of origin", "country"], "where": ["place of birth", "place of death", "place of burial"],
       "buried": ["place of burial"], "performer": ["performer"], "sang": ["performer"], "composer": ["composer"],
       "award": ["award received"], "study": ["educated at"], "studied": ["educated at"], "educated": ["educated at"],
       "work": ["employer"], "founded": ["founded by", "inception"], "founder": ["founded by"], "producer": ["producer"],
       "publisher": ["publisher"], "cause": ["cause of death"], "when": ["date of birth", "date of death", "publication date"]}

def rel_targets(q):
    ql = q.lower(); T = set()
    for k, v in SYN.items():
        if re.search(r"\b" + re.escape(k) + r"\b", ql): T.update(v)
    return T

class Index:
    def __init__(self, edges):
        self.edges = edges
        self.adj = collections.defaultdict(list)
        for i, e in enumerate(edges):
            self.adj[e["h"]].append(i); self.adj[e["t"]].append(i)
        self.name_by_norm = collections.defaultdict(set)
        for v in self.adj: self.name_by_norm[norm(v)].add(v)
        self.nodes = list(self.adj); self.nid = {v: i for i, v in enumerate(self.nodes)}
        self._A = None
    def link(self, question):
        toks = norm(question).split(); found = []
        for n in range(min(12, len(toks)), 0, -1):
            for i in range(len(toks) - n + 1):
                g = " ".join(toks[i:i + n])
                if len(g) < 3 or g in STOP: continue
                if g in self.name_by_norm and not any(i < j2 and i + n > j1 for j1, j2 in found_spans(found)):
                    found.append((i, i + n, g))
        seeds = set()
        for i, j, g in found: seeds.update(self.name_by_norm[g])
        return seeds
    def A(self):
        if self._A is None:
            r, c = [], []
            for e in self.edges:
                a, b = self.nid[e["h"]], self.nid[e["t"]]; r += [a, b]; c += [b, a]
            n = len(self.nodes)
            A = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
            deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1
            self._A = sp.diags(1 / deg) @ A
        return self._A

STOP = {"the", "film", "who", "what", "which", "of", "is", "and", "or", "a", "an", "in", "was", "did", "born", "died", "where", "when",
        "director", "mother", "father", "same", "country", "first", "both", "are", "do", "does", "from", "song", "band", "his", "her",
        "husband", "wife", "films", "nationality", "spouse", "work", "child", "son", "daughter", "people", "person", "place", "date", "death", "birth"}
def found_spans(found): return [(i, j) for i, j, _ in found]

def rel_score(r, T):
    return 1.0 if r in T else 0.0

def retrieve_khop(idx, question, B=30):
    seeds = idx.link(question); T = rel_targets(question)
    scores = {}
    hop1 = set()
    for s in seeds:
        for i in idx.adj.get(s, []):
            sc = 2 + 2 * rel_score(idx.edges[i]["r"], T)
            scores[i] = max(scores.get(i, 0), sc); hop1.add(i)
    for i in list(hop1):
        e = idx.edges[i]; parent = rel_score(e["r"], T)
        for nb in (e["h"], e["t"]):
            if nb in seeds: continue
            for j in idx.adj.get(nb, []):
                if j in hop1: continue
                sc = 0.5 + parent + 2 * rel_score(idx.edges[j]["r"], T)
                scores[j] = max(scores.get(j, 0), sc)
    ranked = sorted(scores, key=lambda i: (-scores[i], i))[:B]
    return ranked, seeds

def retrieve_ppr(idx, question, B=30, alpha=0.5, iters=30):
    seeds = idx.link(question); T = rel_targets(question)
    if not seeds: return [], seeds
    n = len(idx.nodes); p = np.zeros(n)
    for s in seeds: p[idx.nid[s]] = 1
    p /= p.sum(); x = p.copy(); AT = idx.A().T.tocsr()
    for _ in range(iters): x = (1 - alpha) * p + alpha * (AT @ x)
    cand = set()
    # candidate edges: 2-hop neighbourhood of seeds (PPR mass is negligible beyond)
    frontier = set(seeds)
    for _ in range(2):
        nxt = set()
        for v in frontier:
            for i in idx.adj.get(v, []):
                cand.add(i); e = idx.edges[i]; nxt.update([e["h"], e["t"]])
        frontier = nxt
    sc = {i: (x[idx.nid[idx.edges[i]["h"]]] + x[idx.nid[idx.edges[i]["t"]]]) * (1 + 2 * rel_score(idx.edges[i]["r"], T)) for i in cand}
    ranked = sorted(sc, key=lambda i: (-sc[i], i))[:B]
    return ranked, seeds

RETRIEVERS = {"KHOP": retrieve_khop, "PPR": retrieve_ppr}

def gold_support(idx, ranked, qid, n_ev):
    """Fraction and all-flag of the question's gold evidence edges retrieved intact (not corrupted)."""
    hit = set()
    for i in ranked:
        e = idx.edges[i]
        if e["corrupt"]: continue
        for (q, k) in e["gold"]:
            if q == qid: hit.add(k)
    return len(hit) / n_ev, len(hit) == n_ev

def serialize(idx, ranked, provenance):
    lines = []
    for i in ranked:
        e = idx.edges[i]
        s = f"({e['h']}; {e['r']}; {e['t']})"
        if provenance and e["sent"]: s += f' -- source: "{e["sent"].strip()}"'
        lines.append(s)
    return "\n".join(lines)
