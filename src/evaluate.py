"""Milestone 4 — Evaluate the RAG pipeline on the held-out test split.

For each test question we run the RAG pipeline (generator model) to get an
answer + the chunks it retrieved, then grade it with a SEPARATE judge model
(config.JUDGE_MODEL):

  RAGAS (answerable questions only)
    RETRIEVAL   context_precision  — were retrieved chunks relevant?
                context_recall     — did we retrieve everything needed?
    GENERATION  faithfulness       — is the answer grounded (no hallucination)?
                answer_relevancy   — does the answer address the question?
  ABSTENTION (every question)
    The judge says whether the answer declines ("not in the documents").
    Declining is correct for unanswerable questions and wrong for the rest.
    RAGAS can't measure this: answer_relevancy scores any "I don't know" as 0.
  CORRECTNESS (answerable questions)
    The judge compares the answer with the gold answer in two yes/no checks —
    does it contradict it, and is it complete — scored 0 / 0.5 / 1. Unlike the
    two metrics below, it ignores length and wording.
  GOLD SIMILARITY (answerable questions)
    Embedding cosine similarity to the gold answer — the same signal the
    optimizers maximize on the train split, measured here on unseen questions.
    It rewards answers phrased like the gold sentences, so read it with
    correctness and answer length.

Usage:
    python -m src.evaluate                # full test split
    python -m src.evaluate --limit 3      # quick smoke test on 3 questions
    python -m src.evaluate --tag optimized --program models/optimized_rag.json
"""
from __future__ import annotations

import argparse
import json
import time

import dspy
import ollama
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
from .gold import is_answerable, load_gold
from .optimize import answer_similarity
from .rag import RAG, configure_lm

GENERATION = ["faithfulness", "answer_relevancy"]
RETRIEVAL = ["context_precision", "context_recall"]
RAGAS_NAMES = {"llm_context_precision_with_reference": "context_precision"}

ABSTAIN_PROMPT = """Classify the ANSWER given by a policy question-answering assistant.

ANSWERED: the answer states information that responds to the question (even partly, even if the response is "no").
DECLINED: the answer only says the information is unknown, not available, or not covered, without stating the information asked for.

QUESTION: {question}
ANSWER: {answer}

Reply with exactly one word: ANSWERED or DECLINED."""


CORRECT_PROMPT = """Compare an ANSWER with a REFERENCE answer from company policy documents.

QUESTION: {question}
REFERENCE: {reference}
ANSWER: {answer}

1. contradicts: does the ANSWER state any fact that differs from the REFERENCE (a different number, time, limit, approver, place, or yes/no)? Ignore wording and length.
2. complete: does the ANSWER give every fact the QUESTION asks for? A short answer can be complete.

Reply with JSON only: {{"contradicts": true or false, "complete": true or false}}"""


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


def answer_questions(rag: RAG, items: list[dict], threads: int = 2) -> list[dict]:
    """Run the pipeline on every item; one row per question (answer + contexts)."""
    examples = [dspy.Example(question=it["question"]).with_inputs("question") for it in items]
    preds = rag.batch(examples, num_threads=threads, disable_progress_bar=True)
    rows = []
    for it, pred in zip(items, preds):
        rows.append({
            "id": it["id"], "type": it["type"], "question": it["question"],
            "ground_truth": it["ground_truth"],
            "answer": pred.answer if pred is not None else "",
            "contexts": list(pred.context) if pred is not None else [],
            "error": pred is None,
        })
    return rows


def ragas_scores(rows: list[dict], metric_names: list[str]) -> pd.DataFrame:
    """RAGAS scores for the given rows, one row per input row, keyed by id."""
    available = {
        "context_precision": LLMContextPrecisionWithReference,
        "context_recall": LLMContextRecall,
        "faithfulness": Faithfulness,
        "answer_relevancy": AnswerRelevancy,
    }
    dataset = EvaluationDataset([
        SingleTurnSample(user_input=r["question"], response=r["answer"],
                         retrieved_contexts=r["contexts"], reference=r["ground_truth"])
        for r in rows
    ])
    judge_llm, judge_emb = build_judge()
    # A local judge chokes if hit with 16 parallel requests (RAGAS default),
    # so cap concurrency and give each judge call room to finish.
    run_config = RunConfig(timeout=600, max_workers=3, max_retries=3)
    result = evaluate(dataset, metrics=[available[m]() for m in metric_names],
                      llm=judge_llm, embeddings=judge_emb, run_config=run_config,
                      show_progress=False)
    df = result.to_pandas().rename(columns=RAGAS_NAMES)
    df["id"] = [r["id"] for r in rows]
    return df[["id", *metric_names]]


def abstained(question: str, answer: str, client: ollama.Client | None = None) -> int:
    client = client or ollama.Client(host=config.OLLAMA_BASE_URL)
    reply = client.chat(
        model=config.JUDGE_MODEL,
        messages=[{"role": "user", "content": ABSTAIN_PROMPT.format(question=question, answer=answer)}],
        options={"temperature": 0},
    )
    return int(reply["message"]["content"].strip().upper().startswith("DECLINED"))


def correctness(question: str, reference: str, answer: str, client: ollama.Client | None = None) -> float:
    """1 = complete and consistent with the gold answer, 0.5 = consistent but incomplete, 0 = contradicts it."""
    client = client or ollama.Client(host=config.OLLAMA_BASE_URL)
    reply = client.chat(
        model=config.JUDGE_MODEL,
        messages=[{"role": "user", "content": CORRECT_PROMPT.format(
            question=question, reference=reference, answer=answer)}],
        options={"temperature": 0}, format="json",
    )
    try:
        verdict = json.loads(reply["message"]["content"])
    except json.JSONDecodeError:
        return float("nan")
    if verdict.get("contradicts"):
        return 0.0
    return 1.0 if verdict.get("complete") else 0.5


def score_rows(rows: list[dict], with_retrieval: bool = True) -> pd.DataFrame:
    """Per-question scores: RAGAS on answerable rows, abstention + similarity on all."""
    answerable = [r for r in rows if is_answerable(r)]
    metrics = GENERATION + (RETRIEVAL if with_retrieval else [])
    df = pd.DataFrame(rows)
    df = df.merge(ragas_scores(answerable, metrics), on="id", how="left")
    df["gold_similarity"] = [answer_similarity(r["ground_truth"], r["answer"]) for r in rows]
    client = ollama.Client(host=config.OLLAMA_BASE_URL)
    df["abstained"] = [abstained(r["question"], r["answer"], client) for r in rows]
    df["correctness"] = [correctness(r["question"], r["ground_truth"], r["answer"], client)
                         if is_answerable(r) else float("nan") for r in rows]
    df["contexts"] = df["contexts"].map(lambda cs: "\n---\n".join(cs))
    return df


def summarize(df: pd.DataFrame) -> dict:
    """Run-level numbers. RAGAS and similarity are means over answerable questions."""
    ans = df[df["type"] != "unanswerable"]
    out = {m: ans[m].mean() for m in ["correctness_reviewed", "correctness", *GENERATION, *RETRIEVAL,
                                      "gold_similarity"] if m in df}
    for m in ["correctness_reviewed", "correctness"]:
        if m in df:
            out[f"{m}_multi_hop"] = df.loc[df["type"] == "multi_hop", m].mean()
    out["gold_similarity_multi_hop"] = df.loc[df["type"] == "multi_hop", "gold_similarity"].mean()
    out["answer_words"] = ans["answer"].fillna("").str.split().str.len().mean()
    out["abstention_accuracy"] = df.loc[df["type"] == "unanswerable", "abstained"].mean()
    out["false_abstention"] = ans["abstained"].mean()
    out["ragas_nan"] = int(ans[[m for m in GENERATION if m in df]].isna().sum().sum())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only first N questions")
    parser.add_argument("--tag", default="baseline", help="label for the results file")
    parser.add_argument("--program", default=None, help="path to an optimized DSPy program (.json)")
    parser.add_argument("--split", default="test", choices=["test", "train"])
    args = parser.parse_args()

    items = load_gold(split=args.split)[: args.limit]

    print(f"Configuring generator ({config.OLLAMA_MODEL}) and building RAG...")
    configure_lm(cache=False)
    rag = RAG()
    if args.program:
        rag.load(args.program)  # load an optimized program (Milestone 5)
        print(f"Loaded optimized program from {args.program}")

    print(f"Answering {len(items)} {args.split} questions...")
    rows = answer_questions(rag, items)
    print(f"Scoring with judge ({config.JUDGE_MODEL}) — this is the slow part...")
    t = time.time()
    df = score_rows(rows)
    elapsed = time.time() - t

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / f"eval_{args.tag}.csv"
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print(f"RESULTS  [{args.tag}]   ({elapsed:.0f}s judging {len(items)} questions)")
    print("=" * 60)
    for name, value in summarize(df).items():
        print(f"  {name:28s} {value:.3f}")
    print(f"\nPer-question detail saved to: {out_path}")


if __name__ == "__main__":
    main()
