import json, re
import llm
RELS = ["director","date of birth","father","date of death","publication date","country of citizenship","spouse","mother",
        "place of birth","place of death","country of origin","country","performer","sibling","child","inception",
        "award received","composer","educated at","place of burial","employer","publisher","founded by","producer",
        "cause of death","editor","creator","place of detention","occupation","genre","cast member","screenwriter",
        "located in","member of","position held","nationality"]
SYS = "You are a precise information-extraction system that builds a knowledge graph from encyclopedic text."
TMPL = """Extract all factual (subject, relation, object) triples from the paragraph below.

Rules:
- Use full canonical entity names (e.g. the paragraph title for "he", "she", "the film"); resolve pronouns.
- Prefer these relation labels when applicable: {rels}. Other short lowercase labels are allowed.
- Dates as written in the text (e.g. "12 May 1950"). One fact per line.
- Output ONLY lines of the form: sentence_index | subject | relation | object
  where sentence_index is the 0-based index of the sentence stating the fact.

Title: {title}
Sentences:
{sents}
"""
def prompt(title, sents):
    s = "\n".join(f"[{i}] {x.strip()}" for i, x in enumerate(sents))
    return TMPL.format(rels=", ".join(RELS), title=title, sents=s)

def parse(text, n_sents):
    out = []
    for line in text.splitlines():
        parts = [p.strip().strip('`*') for p in line.split("|")]
        if len(parts) != 4: continue
        m = re.match(r"\[?(\d+)\]?", parts[0])
        if not m: continue
        sid = int(m.group(1)); h, r, t = parts[1:]
        if not h or not r or not t or sid >= max(n_sents, 1): continue
        out.append((sid, h, r.lower(), t))
    return out

def extract_all(model, paras, workers=16):
    titles = list(paras)
    res = llm.batch(model, [prompt(t, paras[t]) for t in titles], system=SYS, max_tokens=1500, workers=workers)
    trip = {t: parse(r["text"], len(paras[t])) for t, r in zip(titles, res)}
    errs = sum(1 for r in res if r.get("error"))
    tok = sum(r["in"] + r["out"] for r in res)
    return trip, errs, tok
