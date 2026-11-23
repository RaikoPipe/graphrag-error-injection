# When the Graph Breaks: controlled error injection for GraphRAG

Code, data and results for the paper *When the Graph Breaks: A Controlled Error-Injection Study of How Knowledge-Graph Construction Errors Propagate in GraphRAG*.

The study embeds the gold evidence triples of 600 2WikiMultihopQA questions in an LLM-extracted background graph. It then injects six typed construction errors (entity deletion, edge deletion, entity split, entity merge, relation corruption, spurious edges) at rates of 10–50%, and traces each error through two retrievers, two context formats and five generators.

## Repository layout

| Path | Content |
|---|---|
| `study/kg.py` | Graph construction, gold anchoring, error operators, k-hop and PPR retrievers |
| `study/llm.py` | Cached LLM client (Anthropic API, Ollama cloud) |
| `study/qa.py`, `study/extract.py`, `study/runner.py` | Prompts, answer parsing and scoring, triple extraction, condition runner |
| `study/run_extras.py` | Data preparation, extraction, extractor profiles, natural-graph runs, baselines, judge validation |
| `study/run_grid.py`, `run_retrieval.py`, `run_presence.py`, `run_judge.py` | Main generation grid, retrieval grid, gold-evidence presence, judging |
| `study/analysis.py`, `study/make_figures.py` | Statistics and figures |
| `study/design.md` | Study protocol with deviations log D1–D4 |
| `study/questions.json`, `study/paragraphs.json` | 600 sampled questions and the 3,980-paragraph corpus |
| `study/extract_*.json` | Raw triple extractions of the three LLM extractors |
| `study/results/` | All generated answers (parquet), judgments, analysis tables |
| `study/cache/llm_cache.sqlite` | Cache of all 134k model calls (prompts and responses) |
| `paper/` | LaTeX sources: `main.tex` (ACM TOIS submission), `main_preprint.tex` (neutral preprint), `refs.bib`, figures |

## Reproducing the results

```bash
pip install -r requirements.txt          # Python 3.12
cd study
python run_extras.py prepare             # downloads 2WikiMultihopQA dev split, samples 600 questions (seed 42)
python run_extras.py extract             # triple extraction (gpt-oss-120b, Claude Haiku 4.5, Gemma 4 31B)
python run_retrieval.py                  # deterministic retrieval grid, no LLM calls
python run_presence.py                   # gold-evidence presence per condition
python run_grid.py claude-haiku-4-5-20251001 main 16
python run_grid.py claude-sonnet-4-6 sub03 8
python run_grid.py ollama:gpt-oss:120b sub 8   # likewise ollama:gemma4:31b, ollama:gpt-oss:20b
python run_extras.py baselines natural profiles
python run_judge.py ollama:deepseek-v4.1-flash 8
python run_extras.py judge_validation
python analysis.py
python make_figures.py
```

With the shipped cache, every model call is served locally and the full pipeline reruns without API access. To issue new calls, set the environment variables `CLAUDE` (Anthropic API key) and `OLLAMA` (Ollama cloud key).

Two scripts handle model calls differently:
- `run_judge.py` judges all answers present in `results/`.
- `run_extras.py judge_validation` re-judges the shipped 300-item validation sample; it does not redraw it.

## Data and licences

- **2WikiMultihopQA** (Ho et al., COLING 2020) is distributed under the Apache-2.0 licence ([Alab-NII/2wikimultihop](https://github.com/Alab-NII/2wikimultihop)). `questions.json` and `paragraphs.json` are derived from its development split.
- **Model outputs** in `results/` and `cache/` were generated with Claude Haiku 4.5, Claude Sonnet 4.6, Claude Opus 4.5 (adjudication only), gpt-oss-120b/20b, Gemma 4 31B and DeepSeek-V4.1-Flash. Their reuse is subject to the respective providers' terms.
- **Licence for this repository:** TODO (choose before publishing, e.g. MIT or Apache-2.0 for code, CC BY 4.0 for data).

## Citation

TODO (add after the preprint is posted).
