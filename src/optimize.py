"""Milestone 5 — Make the pipeline self-improve with a DSPy optimizer.

This is the payoff of using DSPy. Rather than hand-editing the prompt, we let
an optimizer search for a better one, scored by a fast embedding-similarity
metric against the gold answers of the TRAIN split only:

  bootstrap  BootstrapFewShot — runs the pipeline over training questions, keeps
             traces whose answer scored well, and inserts them into the prompt
             as few-shot demonstrations.
  mipro      MIPROv2 — also proposes new *instructions* and searches over
             instruction x demo combinations (Bayesian optimization).

Both take a seed, so the experiment can measure how much the result depends on
the optimizer's random choices rather than on the method itself.

Usage:
    python -m src.optimize                          # BootstrapFewShot, seed 0
    python -m src.optimize --optimizer mipro --seed 3
"""
from __future__ import annotations

import argparse
import random

import dspy
from sentence_transformers import SentenceTransformer, util

from . import config
from .gold import load_gold
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
    score = answer_similarity(example.answer, prediction.answer)
    if trace is not None:          # optimizer mode -> boolean gate
        return score >= 0.60
    return score                   # evaluation mode -> raw number


def answer_similarity(gold: str, answer: str) -> float:
    g = _embedder.encode(gold, convert_to_tensor=True)
    a = _embedder.encode(answer or "", convert_to_tensor=True)
    return float(util.cos_sim(g, a).item())


def load_trainset(seed: int = 0) -> list[dspy.Example]:
    """The train split as DSPy Examples, in a seed-dependent order.

    BootstrapFewShot keeps the first training traces that pass the metric, so
    the order decides which demonstrations end up in the prompt.
    """
    train = [
        dspy.Example(question=it["question"], answer=it["ground_truth"]).with_inputs("question")
        for it in load_gold(split="train")
    ]
    random.Random(seed).shuffle(train)
    return train


def compile_program(optimizer: str = "bootstrap", seed: int = 0) -> RAG:
    """Compile a RAG program from the train split. Needs configure_lm() first."""
    train = load_trainset(seed)
    if optimizer == "bootstrap":
        opt = dspy.BootstrapFewShot(
            metric=similarity_metric,
            max_bootstrapped_demos=2,   # up to 2 self-generated examples in the prompt
            max_labeled_demos=2,        # up to 2 straight from the trainset
        )
        return opt.compile(RAG(), trainset=train)
    if optimizer == "mipro":
        # The generator also proposes the instructions, so the judge model never
        # touches the system it grades. "light" keeps the search laptop-sized.
        opt = dspy.MIPROv2(
            metric=similarity_metric,
            auto="light",
            max_bootstrapped_demos=2,
            max_labeled_demos=2,
            num_threads=2,
            seed=seed,
            verbose=False,
        )
        return opt.compile(RAG(), trainset=train, seed=seed, requires_permission_to_run=False)
    raise ValueError(f"unknown optimizer: {optimizer}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", choices=["bootstrap", "mipro"], default="bootstrap")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(OPTIMIZED_PATH))
    args = parser.parse_args()

    configure_lm()
    print(f"Optimizing RAG with {args.optimizer} (seed {args.seed}) on the train split...")
    optimized = compile_program(args.optimizer, args.seed)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    optimized.save(args.out)
    print(f"Saved optimized program to {args.out}")
    print(f"Now compare:  python -m src.evaluate --tag optimized --program {args.out}")


if __name__ == "__main__":
    main()
