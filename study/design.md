# Pre-registration: Controlled error injection in GraphRAG knowledge-graph construction

Version 1.0, frozen before any outcome data (Oct 2026). Later deviations are appended in §9 with a date and reason.

## 1. Gap

GraphRAG error modes at the construction stage (coverage loss, entity split/merge, noisy or false triples) have been described and correlated with end-task accuracy, but never isolated causally. No study varies typed construction errors at controlled rates against a gold graph and traces how they propagate through retrieval to generation (review gaps G1/G2).

## 2. Research questions

- **RQ1 Dose–response.** How does answer quality change as each typed construction error increases in rate?
- **RQ2 Stage dominance.** How much of the variance in answer quality is due to graph quality, compared with retriever choice, provenance attachment and generator choice?
- **RQ3 Mechanism.** Which share of each error type's damage arises in retrieval (gold evidence no longer retrieved) and which in generation (evidence retrieved, answer still wrong)?
- **RQ4 Mitigation and moderation.** Do provenance text, stronger generators and question type moderate the damage?
- **RQ5 External validity.** Where do real LLM extractors sit on the injected dose–response curves, and do the curves predict end-to-end accuracy on naturally extracted graphs?

## 3. Hypotheses (directional, falsifiable)

- **H1.** Accuracy decreases monotonically with error rate for every operator. Omission operators (entity deletion, edge deletion, split) reduce gold-support recall more steeply than commission operators (merge, relation corruption, spurious edges).
- **H2.** Within the realistic range (rate ≤ 0.3), the graph-error factor (operator × rate) explains more variance in accuracy than retriever choice, provenance attachment or generator choice, each taken separately.
- **H3.** For split errors, more than 50% of the accuracy loss is mediated by retrieval (loss of gold support). For relation corruption and merge errors, more than 50% of the loss occurs in generation: gold support is present or replaced, and the answer is wrong.
- **H4.** Commission errors produce higher error-adoption rates (the answer equals an injected false value) and lower abstention rates than omission errors. Omission errors shift outcomes toward abstention.
- **H5.** Attaching source sentences (provenance) to triples reduces error adoption under commission errors (operator × provenance interaction). It does not reduce the loss caused by omission errors.
- **H6.** Stronger generators reduce generation-stage loss but not retrieval-stage loss. The accuracy gap between generators therefore narrows under omission errors and stays the same or widens under commission errors.
- **H7 (exploratory).** Accuracy on a graph built entirely by an LLM extractor falls within the 95% prediction interval of a model fitted on the injected curves, given the extractor's measured error profile.

## 4. Data and graph

- **Source.** 2WikiMultihopQA dev split (Ho et al., 2020). Each question has gold evidence triples, supporting sentences and 10 context paragraphs, of which 8 are distractors.
- **Sample.** N questions stratified across the four types (compositional, inference, comparison, bridge-comparison). The pilot sets N, with a target of ≥ 400.
- **Corpus.** The union of all context paragraphs of the sampled questions (shared pool; distractors from other questions are retrievable).
- **Gold-anchored graph G\*.**
  - A background graph contains triples that an LLM extractor produced from all corpus paragraphs. Its entity names are normalised and fuzzy-resolved to gold entity names.
  - Background triples that duplicate a gold fact (same head and tail after resolution) are removed, so that each gold fact exists exactly once.
  - All gold evidence triples are then added, each with its supporting sentence as provenance. This gives full experimental control over each gold fact, while the background graph supplies realistic distractors and hubs.

## 5. Factors

| Factor | Levels |
|---|---|
| Error operator | ENT_DEL, EDGE_DEL, SPLIT, MERGE, REL_CORRUPT, SPURIOUS |
| Rate p | 0 (clean), 0.1, 0.2, 0.3, 0.5 |
| Injection seed | 2 for the retrieval grid; 1 for the generation grid (question-level variation dominates) |
| Retriever | KHOP (entity-linked 2-hop expansion, lexical relation ranking), PPR (personalised PageRank from linked seeds; HippoRAG-style) |
| Provenance | triples only / triples + source sentence |
| Generator | primary model on the full grid; ≥ 2 other models of different families and sizes on a subset grid |
| References | closed-book (no retrieval); dense passage RAG (TF-IDF top-5 over the corpus) |

Operator definitions apply to the whole graph, sampled uniformly:

- **ENT_DEL** removes p·|V| nodes and their incident edges.
- **EDGE_DEL** removes p·|E| edges.
- **SPLIT** gives p·|V| nodes an alias copy (surface variant: abbreviation, diacritics stripped, or a token dropped). Each incident edge moves to the alias with probability 0.5.
- **MERGE** fuses p·|V| nodes with their most name-similar other node. The surviving name is chosen at random and edges are unioned.
- **REL_CORRUPT** replaces the tail of p·|E| edges with the tail of a random other edge that has the same relation label (a plausible false fact). The provenance sentence stays the original one.
- **SPURIOUS** adds p·|E| edges whose head is random and whose relation and tail are sampled jointly from existing edges. The provenance is a random sentence of the head's source paragraph.

Retrieval budget: top B = 30 triples (fixed). Every injection logs whether it touched a gold triple or gold node of each question.

## 6. Outcomes

- **Primary.** Judged accuracy: a single-answer LLM judge compares the answer with the gold answer and aliases. The judge is from a different model family than the primary generator. Judging is single-answer only (no pairwise comparison, so no position bias), with temperature 0, and is validated on a manually checked sample of ≥ 100 items.
- **Secondary.** Exact match (EM) and token F1; abstention rate (the model outputs "insufficient information"); error-adoption rate (the answer matches an injected false value that is present in the context); gold-support recall (all gold triples of the question retrieved intact); context tokens.

## 7. Analysis plan

1. **Dose–response.** Mean accuracy per operator × rate with question-cluster bootstrap 95% CIs (2,000 resamples). Logistic mixed model `correct ~ rate × operator + retriever + provenance + (1 | question)`. H1 holds if all rate slopes are < 0 (Holm-corrected) and the omission slopes on gold-support recall are steeper than the commission slopes.
2. **Variance decomposition (H2).** Shapley/ANOVA decomposition of the explained variance in condition-level accuracy over the factors graph-error, retriever, provenance and generator, restricted to p ≤ 0.3. H2 holds if graph-error has the largest share.
3. **Mediation (H3).** Per question, the drop in P(correct) from the clean graph is decomposed into a retrieval term (change in gold-support probability × accuracy given support) and a generation term (change in accuracy given support status). Shares come with bootstrap CIs.
4. **Outcome shift (H4).** Multinomial outcome (correct / wrong / abstain / adopted) by operator class, tested with a χ² test and mixed-model contrasts.
5. **Moderation (H5, H6).** Interaction terms operator class × provenance and operator class × generator in the mixed model; Wald tests with Holm correction.
6. **External validity (H7).** Measure the natural error profile of LLM extractors (coverage, split, merge and relation-error rates relative to gold). Predict accuracy from the injected model and compare with the observed end-to-end accuracy on the extractor-only graph.

Significance level α = 0.05, two-sided; family-wise Holm correction within each hypothesis.

## 8. Threats anticipated

- 2Wiki is Wikipedia-based, so the parametric-knowledge confound is estimated with the closed-book reference.
- Synthetic operators may not match natural error distributions; RQ5 tests this.
- LLM-judge error is estimated through manual validation.
- A single dataset limits generality; this is stated as a limitation.

## 9. Deviations log

**D1 (pilot, 2026-10-14). Final design after the pilot** (60 questions, 14 conditions, 3 generators; ~11 calls/s on Anthropic and ~4.5 calls/s on Ollama-cloud).
- N = 600 questions (150 per type). The pilot effects were large (clean EM ≈ 0.85; EDGE_DEL at 0.3 ≈ 0.30), so the precision of N = 600 is ample.
- Background extractor: gpt-oss-120b (Ollama-cloud).
- Primary generator: claude-haiku-4-5, on the full grid (25 graph conditions × 2 retrievers × 2 provenance levels).
- Secondary generators: gpt-oss-120b, gemma4-31b and gpt-oss-20b (KHOP retriever; clean graph and rates 0.1/0.3; both provenance levels); claude-sonnet-4-6 (KHOP; clean graph and rate 0.3).
- Judge: deepseek-v4.1-flash, which is not used as a generator; claude-sonnet-4-6 acts as a second judge on a validation sample.
- Retrieval grid: rates {0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5} × 3 seeds.
- Excluded model: nemotron-3-nano (empty outputs in the smoke test).

**D2. Sensitivity subset.** The 2WikiMultihopQA gold evidence contains naming inconsistencies (e.g. "Ric Flair" vs. "Richard Fliehr"), which cause some gold-support failures even on the clean graph. Primary analyses keep all 600 questions as pre-registered. A sensitivity analysis is added on the "clean-answerable" subset, i.e. the questions with full gold support under KHOP on the clean graph.

**D3. Human validation of the judge.** The plan called for a manually checked sample of 100 judge decisions. No human annotator was available in this run, so the judge is validated by agreement with a second LLM judge, and disagreements are adjudicated by a stronger third model (claude-opus-4-5). A human check by the authors remains to be done and is flagged in the paper.

**D4. Estimation (analysis).** The confirmatory models are fitted as population-averaged logistic GEE models with an exchangeable working correlation within questions, instead of the pre-registered GLMM with question random intercepts. The GEE gives cluster-robust inference at much lower computational cost on 123k rows; the fixed-effect structure is as pre-registered. The mediation decomposition uses "gold slot retrieved", i.e. all gold evidence edges are in the context, whether corrupted or not. Under this definition, retrieval loss and generation loss can be separated for REL_CORRUPT.
