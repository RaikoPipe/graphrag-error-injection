"""Answer generation, parsing, lexical scoring and single-answer LLM judging."""
import re, string, collections
import llm

SYS_GEN = "You answer questions using a knowledge graph. Be concise."
TMPL_GEN = """Knowledge graph facts (subject; relation; object):
{ctx}

Question: {q}

Answer based on the facts above. If they do not contain enough information, reply exactly: insufficient information.
Respond with a single final line of the form "Answer: <short answer>" (an entity name, a date, or yes/no)."""
TMPL_CB = """Question: {q}

Respond with a single final line of the form "Answer: <short answer>" (an entity name, a date, or yes/no). If you do not know, reply "Answer: insufficient information"."""
TMPL_DENSE = """Retrieved passages:
{ctx}

Question: {q}

Answer based on the passages above. If they do not contain enough information, reply exactly: insufficient information.
Respond with a single final line of the form "Answer: <short answer>" (an entity name, a date, or yes/no)."""

def gen_prompt(ctx, q): return TMPL_GEN.format(ctx=ctx if ctx.strip() else "(no facts retrieved)", q=q)

def parse_answer(text):
    m = re.findall(r"answer\s*[:：]\s*(.+)", text, flags=re.I)
    a = (m[-1] if m else text.strip().splitlines()[-1] if text.strip() else "").strip().strip("*").strip()
    return a.rstrip(".")

def is_abstain(a): return "insufficient information" in a.lower() or a.strip() == ""

def _norm(s):
    s = s.lower(); s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())
def em(pred, gold): return float(_norm(pred) == _norm(gold))
def f1(pred, gold):
    p, g = _norm(pred).split(), _norm(gold).split()
    c = sum((collections.Counter(p) & collections.Counter(g)).values())
    if c == 0: return 0.0
    pr, rc = c / len(p), c / len(g); return 2 * pr * rc / (pr + rc)

SYS_J = "You are a strict, consistent grader of short answers."
TMPL_J = """Question: {q}
Gold answer: {g}
Predicted answer: {p}

Is the predicted answer semantically equivalent to the gold answer (same entity, same date/year, or same yes/no)? Minor formatting or partial-name differences that clearly refer to the same entity count as correct. An abstention is incorrect.
Reply with exactly one word: CORRECT or INCORRECT."""

def judge(model, items, workers=16):
    """items: list of (question, gold, pred). Returns list of 0/1 (None on failure). EM shortcut first."""
    out = [None] * len(items); todo = []
    for i, (q, g, p) in enumerate(items):
        if is_abstain(p): out[i] = 0
        elif em(p, g): out[i] = 1
        else: todo.append(i)
    res = llm.batch(model, [TMPL_J.format(q=items[i][0], g=items[i][1], p=items[i][2]) for i in todo],
                    system=SYS_J, max_tokens=400, workers=workers)
    for i, r in zip(todo, res):
        t = r["text"].upper()
        out[i] = 1 if "CORRECT" in t and "INCORRECT" not in t else (0 if "INCORRECT" in t else None)
    return out
