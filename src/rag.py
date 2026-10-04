"""Milestone 3 — The RAG module, written in DSPy.

The DSPy idea in one paragraph:
    Instead of writing a prompt string like "You are a helpful assistant.
    Given the context: {context} answer: {question}", you DECLARE the task as a
    typed Signature (inputs -> outputs with descriptions). DSPy turns that
    declaration into an actual prompt at runtime. Later, a DSPy *optimizer* can
    rewrite that prompt — add instructions, pick few-shot examples — by testing
    candidates against a metric. You optimize behavior with data, not by hand.
"""
from __future__ import annotations

import dspy

from . import config
from .retriever import ChromaRetriever


def configure_lm(cache: bool = True) -> dspy.LM:
    """Point DSPy at our local Ollama server. No API keys, fully offline.

    'ollama_chat/<model>' tells litellm (which DSPy uses under the hood) to use
    Ollama's chat endpoint. temperature=0 makes answers deterministic, which is
    important so our evaluation numbers are stable and comparable run-to-run.
    cache=False makes every call hit the model, so repeated runs are real
    re-runs rather than answers replayed from DSPy's cache.
    """
    lm = dspy.LM(
        model=f"ollama_chat/{config.OLLAMA_MODEL}",
        api_base=config.OLLAMA_BASE_URL,
        api_key="",            # Ollama needs none
        temperature=0.0,
        max_tokens=512,
        cache=cache,
    )
    dspy.configure(lm=lm)
    return lm


class GenerateAnswer(dspy.Signature):
    """Answer the question using ONLY the provided context.

    If the context does not contain the answer, reply that you don't know.
    Do not use outside knowledge and do not invent facts, numbers, or policies.
    """

    context: str = dspy.InputField(desc="relevant excerpts from company policy documents")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="a concise answer grounded in the context")


class RAG(dspy.Module):
    """Retrieve-then-generate. This is the whole pipeline as one callable."""

    def __init__(self, retriever: ChromaRetriever | None = None, k: int = config.TOP_K):
        super().__init__()
        self.retriever = retriever or ChromaRetriever()
        self.k = k
        # ChainOfThought asks the model to reason step-by-step before answering.
        # DSPy will treat this module's instructions/examples as optimizable.
        self.generate = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question: str) -> dspy.Prediction:
        chunks = self.retriever(question, k=self.k)
        context = "\n\n".join(chunks)
        prediction = self.generate(context=context, question=question)
        # Attach the retrieved context so we can evaluate retrieval quality later.
        return dspy.Prediction(answer=prediction.answer, context=chunks)


if __name__ == "__main__":
    # Quick manual smoke test.
    configure_lm()
    rag = RAG()
    q = "How many days of annual leave do full-time employees get?"
    pred = rag(question=q)
    print("Q:", q)
    print("A:", pred.answer)
