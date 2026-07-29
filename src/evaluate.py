"""Milestone 4 — Evaluate the RAG pipeline with RAGAS.

For each gold question we run the RAG pipeline (generator model) to get an
answer + the chunks it retrieved, then hand everything to RAGAS, which uses a
SEPARATE judge model (config.JUDGE_MODEL) to score four metrics:

  RETRIEVAL   context_precision  — were retrieved chunks relevant?
              context_recall     — did we retrieve everything needed?
  GENERATION  faithfulness       — is the answer grounded (no hallucination)?
              answer_relevancy   — does the answer address the question?

Usage:
    python -m src.evaluate                # full gold set
    python -m src.evaluate --limit 3      # quick smoke test on 3 questions
    python -m src.evaluate --tag baseline # label the saved results file
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

from . import config
from .rag import RAG, configure_lm


def build_judge():
    """The RAGAS judge: a stricter model + a local embedding model.

    Ollama speaks the OpenAI API at /v1, so we reuse langchain's ChatOpenAI and
    just point base_url at Ollama. api_key is a dummy — Ollama ignores it.
    """
    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=config.JUDGE_MODEL,
            base_url=f"{config.OLLAMA_BASE_URL}/v1",
            api_key="ollama",
            temperature=0.0,
        )
    )
    judge_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)
    )
    return judge_llm, judge_emb


def run_pipeline_over_goldset(rag: RAG, gold: list[dict]) -> EvaluationDataset:
    """Run RAG on each question and package results the way RAGAS expects."""
    samples = []
    for i, item in enumerate(gold, 1):
        pred = rag(question=item["question"])
        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                response=pred.answer,
                retrieved_contexts=list(pred.context),
                reference=item["ground_truth"],
            )
        )
        print(f"  [{i}/{len(gold)}] answered: {item['question'][:60]}...")
    return EvaluationDataset(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only first N questions")
    parser.add_argument("--tag", default="baseline", help="label for the results file")
    parser.add_argument("--program", default=None, help="path to an optimized DSPy program (.json)")
    args = parser.parse_args()

    gold = json.loads(config.GOLD_QA_PATH.read_text())
    if args.limit:
        gold = gold[: args.limit]

    print(f"Configuring generator ({config.OLLAMA_MODEL}) and building RAG...")
    configure_lm()
    rag = RAG()
    if args.program:
        rag.load(args.program)  # load an optimized program (Milestone 5)
        print(f"Loaded optimized program from {args.program}")

    print(f"Running pipeline over {len(gold)} questions...")
    dataset = run_pipeline_over_goldset(rag, gold)

    print(f"Scoring with RAGAS judge ({config.JUDGE_MODEL}) — this is the slow part...")
    judge_llm, judge_emb = build_judge()
    metrics = [
        LLMContextPrecisionWithReference(),  # retrieval: precision
        LLMContextRecall(),                  # retrieval: recall
        Faithfulness(),                      # generation: grounded?
        AnswerRelevancy(),                   # generation: on-topic?
    ]

    # A local 3B judge chokes if hit with 16 parallel requests (RAGAS default),
    # so cap concurrency and give each judge call room to finish.
    run_config = RunConfig(timeout=300, max_workers=3, max_retries=3)

    t = time.time()
    result = evaluate(dataset, metrics=metrics, llm=judge_llm,
                      embeddings=judge_emb, run_config=run_config)
    elapsed = time.time() - t

    df = result.to_pandas()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / f"eval_{args.tag}.csv"
    df.to_csv(out_path, index=False)

    # Aggregate scores (mean of each metric column).
    metric_cols = [c for c in df.columns if c in
                   {"llm_context_precision_with_reference", "context_recall",
                    "faithfulness", "answer_relevancy"}]
    print("\n" + "=" * 60)
    print(f"RESULTS  [{args.tag}]   ({elapsed:.0f}s for {len(gold)} questions)")
    print("=" * 60)
    for col in metric_cols:
        print(f"  {col:38s} {df[col].mean():.3f}")
    print(f"\nPer-question detail saved to: {out_path}")


if __name__ == "__main__":
    main()
