"""Milestone 5 — Make the pipeline self-improve with a DSPy optimizer.

This is the payoff of using DSPy. Rather than hand-editing the prompt, we let
DSPy's BootstrapFewShot optimizer:
  1. Run the current RAG pipeline over training questions.
  2. Keep the (context, question -> answer) traces where the answer scored well
     against the gold answer (our fast embedding-similarity metric).
  3. Insert those successful traces into the prompt as few-shot demonstrations.
The compiled program is a better prompt, discovered from DATA — not by us.

Usage:
    python -m src.optimize                 # compile and save optimized program
"""
from __future__ import annotations

import json

import dspy
from sentence_transformers import SentenceTransformer, util

from . import config
from .rag import RAG, configure_lm

MODELS_DIR = config.ROOT / "models"
OPTIMIZED_PATH = MODELS_DIR / "optimized_rag.json"

# One embedding model, loaded once, for the cheap similarity metric.
_embedder = SentenceTransformer(config.EMBED_MODEL)


def similarity_metric(example: dspy.Example, prediction, trace=None) -> bool | float:
    """Score a prediction against the gold answer by embedding cosine similarity.

    DSPy calls this a lot, so it must be fast — hence embeddings, not an LLM.
    During optimization (trace is not None) it must return a pass/fail bool so
    BootstrapFewShot can decide whether to keep a demonstration; during plain
    evaluation it returns the raw score.
    """
    gold = _embedder.encode(example.answer, convert_to_tensor=True)
    pred = _embedder.encode(prediction.answer, convert_to_tensor=True)
    score = float(util.cos_sim(gold, pred).item())
    if trace is not None:          # optimizer mode -> boolean gate
        return score >= 0.60
    return score                   # evaluation mode -> raw number


def load_trainset() -> list[dspy.Example]:
    """Turn the gold Q&A into DSPy Examples, marking `question` as the input."""
    gold = json.loads(config.GOLD_QA_PATH.read_text())
    return [
        dspy.Example(question=item["question"], answer=item["ground_truth"]).with_inputs("question")
        for item in gold
    ]


def main():
    configure_lm()
    trainset = load_trainset()
    # Small hold-out convention: train on the first 8, leave the rest for the
    # honest RAGAS comparison in Milestone 4 (train/test hygiene still applies
    # even when you're optimizing prompts rather than weights).
    train = trainset[:8]

    print(f"Optimizing RAG with BootstrapFewShot on {len(train)} examples...")
    optimizer = dspy.BootstrapFewShot(
        metric=similarity_metric,
        max_bootstrapped_demos=2,   # up to 2 self-generated examples in the prompt
        max_labeled_demos=2,        # up to 2 straight from the trainset
    )
    optimized = optimizer.compile(RAG(), trainset=train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    optimized.save(str(OPTIMIZED_PATH))
    print(f"Saved optimized program to {OPTIMIZED_PATH}")
    print("Now compare:  python -m src.evaluate --tag optimized "
          f"--program {OPTIMIZED_PATH}")


if __name__ == "__main__":
    main()
