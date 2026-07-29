"""Interactive CLI — ask the knowledge base a question.

    python -m src.ask                       # naive RAG, interactive loop
    python -m src.ask "how much parental leave?"   # one-shot
    python -m src.ask --program models/optimized_rag.json   # use optimized RAG

Shows the answer AND the sources it retrieved, so answers are auditable — an
important property for enterprise RAG (users must be able to check the citation).
"""
from __future__ import annotations

import argparse

from .rag import RAG, configure_lm


def answer_once(rag: RAG, question: str) -> None:
    pred = rag(question=question)
    print(f"\nQ: {question}")
    print(f"A: {pred.answer}\n")
    print("Sources (retrieved chunks):")
    for i, chunk in enumerate(pred.context, 1):
        print(f"  [{i}] {chunk[:110].strip()}...")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*", help="a one-shot question")
    parser.add_argument("--program", default=None, help="path to an optimized DSPy program")
    args = parser.parse_args()

    configure_lm()
    rag = RAG()
    if args.program:
        rag.load(args.program)
        print(f"(using optimized program: {args.program})")

    if args.question:
        answer_once(rag, " ".join(args.question))
        return

    print("Acme Robotics knowledge base. Ask a question (Ctrl-C or 'quit' to exit).")
    try:
        while True:
            q = input("\n> ").strip()
            if q.lower() in {"quit", "exit", ""}:
                break
            answer_once(rag, q)
    except (KeyboardInterrupt, EOFError):
        print("\nbye")


if __name__ == "__main__":
    main()
