# Self-Improving, Evaluated RAG (DSPy · ChromaDB · Ollama · RAGAS)

A **local-first** Retrieval-Augmented Generation pipeline over an enterprise
knowledge base that **optimizes its own prompts** (DSPy) and **grades its own
outputs** (RAGAS) — running entirely on a laptop, no cloud API, no keys.

> Built as a portfolio project for graduate ML applications. The goal is to
> demonstrate *programmatic* prompt optimization and *statistical* evaluation
> of an LLM system — not artisanal prompt tweaking.

## Why this design

| Choice | Reason |
|---|---|
| **DSPy** for the pipeline | Declare the task as a typed `Signature`; an optimizer *compiles* the prompt from data instead of hand-tuning strings. |
| **RAGAS** for evaluation | Turns "looks good" into numbers: faithfulness, answer relevancy, context precision/recall. |
| **Separate generator & judge models** | Answering and *grading* are different jobs. The generator (`llama3.2:3b`) is optimized for fluent answers; the judge (`qwen2.5:3b`) is chosen for disciplined, parseable JSON verdicts. |
| **ChromaDB + local embeddings** | Vector search with `all-MiniLM-L6-v2`, persisted to disk, fully offline. |
| **Ollama** (not vLLM) | vLLM needs an NVIDIA GPU/CUDA. Ollama is the Apple-Silicon-native way to serve local LLMs. |

## Architecture

```
INDEXING (once)          data/*.md ──chunk──▶ embed ──▶ ChromaDB
                                                            │
QUERYING                 question ──embed──▶ top-k search ──┘
                                    │
                                    ▼
                         DSPy RAG module (Signature: context, question → answer)
                                    │
                                    ▼
                                 answer + retrieved sources
                                    │
EVALUATION               RAGAS judge (qwen2.5:3b) ──▶ 4 metrics
                                    │
OPTIMIZATION             DSPy BootstrapFewShot ──▶ better prompt (compiled from data)
```

## Setup

```bash
# 1. Install Ollama and pull the models
brew install ollama
ollama serve &                 # start the local model server
ollama pull llama3.2:3b        # generator
ollama pull qwen2.5:3b         # judge

# 2. Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run it

```bash
# Build the vector index from data/*.md
python -m src.ingest

# Ask a question (naive RAG)
python -m src.ask "How much parental leave does the primary caregiver get?"

# Evaluate the baseline pipeline with RAGAS
python -m src.evaluate --tag baseline

# Optimize the prompt from data, then re-evaluate to see the lift
python -m src.optimize
python -m src.evaluate --tag optimized --program models/optimized_rag.json
```

## Results

RAGAS scores on a 12-question gold set over the Acme Robotics policy corpus
(0–1, higher is better). Generator `llama3.2:3b`, judge `qwen2.5:3b`, single run.

| Metric | Half of RAG | Baseline | Optimized | Δ |
|---|---|---|---|---|
| Context Precision | Retrieval | 0.833 | 0.833 | — |
| Context Recall | Retrieval | 1.000 | 1.000 | — |
| Faithfulness | Generation | 0.771 | 0.681 | −0.09 |
| Answer Relevancy | Generation | 0.722 | 0.814 | +0.09 |

**Reading the result (this matters more than the numbers):**

- **Retrieval metrics are unchanged — by design.** BootstrapFewShot optimized only
  the *generation* prompt, never retrieval, so context precision/recall are
  identical. The experiment isolates exactly one variable.
- **The optimizer traded faithfulness for relevancy.** The 2 bootstrapped
  demonstrations made answers more complete and on-topic (relevancy ↑) but also
  more elaborative, so the answer occasionally drifted from what the context
  strictly supports (faithfulness ↓). This is a known RAG tension, observed here
  directly rather than assumed.
- **±0.09 is within run-to-run noise** given 8 training examples, 2 demos, a 3B
  generator and a 3B judge scoring with a single sample. A defensible claim of
  improvement needs multiple seeds and variance estimates — see next steps.

## What I would do next (roadmap)

1. **Reduce metric variance** — run each config 3–5× and report mean ± std;
   raise the judge's self-consistency samples.
2. **Stronger optimizer** — swap BootstrapFewShot for `MIPROv2`, which also
   rewrites the *instructions*, not just the few-shot examples.
3. **Improve retrieval** — semantic/heading-aware chunking instead of fixed
   character windows; try a reranker over the top-k.
4. **Bigger, harder eval set** — multi-hop questions that span documents, plus
   negative questions the corpus can't answer (to test "I don't know").
5. **Stronger judge** — a 7B+ local judge or a frontier API model to tighten
   the metrics (one line in `config.py`).

## Project layout

```
data/            # the knowledge base (fictional company policy .md files)
eval/qa_gold.json    # gold question/answer pairs for evaluation
src/config.py    # all tunable settings (models, chunking, top-k)
src/ingest.py    # chunk + embed + store into ChromaDB
src/retriever.py # ChromaDB-backed semantic search
src/rag.py       # DSPy Signature + RAG module (retrieve → generate)
src/evaluate.py  # RAGAS evaluation with a separate judge model
src/optimize.py  # DSPy BootstrapFewShot prompt optimization
src/ask.py       # interactive CLI
```

## Honest limitations

- **The corpus is fictional and small** by design, so ground-truth answers are
  known and the local model can't have memorized them — every correct answer
  must come from retrieval.
- **The judge is a small local model.** A 3B judge is far more reliable than it
  sounds once you separate it from generation, but a larger judge (or a frontier
  API model) would give lower-variance metrics. The judge is one line in
  `config.py` to swap.
- **RAGAS metrics are estimates**, not ground truth — they are LLM judgments and
  carry the judge's noise. Treat them as a directional signal.
