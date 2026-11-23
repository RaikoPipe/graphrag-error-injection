"""Unified, cached, rate-limited LLM client for Anthropic and Ollama-cloud models."""
import os, json, time, hashlib, sqlite3, threading, random
import requests
from concurrent.futures import ThreadPoolExecutor

CACHE = os.environ.get("LLM_CACHE", "study/cache/llm_cache.sqlite")
os.makedirs(os.path.dirname(CACHE), exist_ok=True)
_lock = threading.Lock()
_conn = sqlite3.connect(CACHE, check_same_thread=False, timeout=60)
_conn.execute("PRAGMA journal_mode=WAL")
_conn.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
_conn.commit()

def _key(model, system, prompt, max_tokens, temperature):
    return hashlib.sha256(json.dumps([model, system, prompt, max_tokens, temperature]).encode()).hexdigest()

def _get(k):
    with _lock:
        r = _conn.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
    return json.loads(r[0]) if r else None

def _put(k, v):
    with _lock:
        _conn.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (k, json.dumps(v)))
        _conn.commit()

def is_ollama(model): return model.startswith("ollama:")

def _call_anthropic(model, system, prompt, max_tokens, temperature):
    body = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]}
    if system: body["system"] = system
    r = requests.post("https://api.anthropic.com/v1/messages", json=body, timeout=180,
                      headers={"x-api-key": os.environ["CLAUDE"], "anthropic-version": "2023-06-01"})
    if r.status_code != 200: raise RuntimeError(f"{r.status_code} {r.text[:300]}")
    j = r.json()
    txt = "".join(b.get("text", "") for b in j["content"] if b["type"] == "text")
    return {"text": txt, "in": j["usage"]["input_tokens"], "out": j["usage"]["output_tokens"]}

def _call_ollama(model, system, prompt, max_tokens, temperature):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = {"model": model.split(":", 1)[1], "messages": msgs, "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens}}
    if "gpt-oss" in model: body["think"] = "low"
    elif "deepseek" in model or "kimi" in model or "glm" in model: body["think"] = False
    r = requests.post("https://ollama.com/api/chat", json=body, timeout=300,
                      headers={"Authorization": f"Bearer {os.environ['OLLAMA']}"})
    if r.status_code != 200: raise RuntimeError(f"{r.status_code} {r.text[:300]}")
    j = r.json()
    return {"text": j["message"].get("content", ""), "in": j.get("prompt_eval_count", 0), "out": j.get("eval_count", 0)}

def call(model, prompt, system="", max_tokens=256, temperature=0.0, retries=6):
    k = _key(model, system, prompt, max_tokens, temperature)
    v = _get(k)
    if v is not None and v.get("text", "").strip(): return v
    fn = _call_ollama if is_ollama(model) else _call_anthropic
    err = None
    for a in range(retries):
        try:
            v = fn(model, system, prompt, max_tokens, temperature)
            if not v["text"].strip(): raise RuntimeError("empty output")
            _put(k, v); return v
        except Exception as e:
            err = e; time.sleep(min(60, 2 ** a + random.random() * 2))
    return {"text": "", "in": 0, "out": 0, "error": str(err)[:300]}

def batch(model, prompts, system="", max_tokens=256, temperature=0.0, workers=8):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda p: call(model, p, system, max_tokens, temperature), prompts))
