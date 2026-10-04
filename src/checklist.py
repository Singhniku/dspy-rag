"""CheckList-style behavioral tests for the RAG pipeline (Ribeiro et al., ACL 2020).

Aggregate scores like RAGAS faithfulness average over questions and can hide
systematic failures. CheckList instead perturbs inputs in controlled ways and
checks whether the output changes the way it should. Three test types, all in
eval/checklist.json:

  invariance   (INV) 15 factual questions, each rephrased 3 ways: synonyms,
               restructured wording, and a casual version with typos. The key
               fact in the answer (e.g. "20 days") must not change. Checked
               with keyword rules, so no judge noise. A test fails when a
               rephrasing gets the fact right and the original doesn't, or
               the other way round.
  directional  (DIR) 16 yes/no questions, each paired with a negated version
               ("allowed" -> "prohibited", "required" -> "not required"). The
               correct answer flips; a test fails when the negated question
               gets the wrong yes/no.
  abstention   (ABS) 18 questions the documents don't answer: 12 near-misses
               that resemble a covered fact (meal allowance in South America,
               when only North America and Europe are given) and 6 off-topic
               ones. A test fails when the answer doesn't decline.

Programs: the baseline plus every compiled program from src/experiment.py
(BootstrapFewShot and MIPROv2, seeds 1-5).

Phases, each saved under eval/results/checklist/ and skipped when done:
  1. answer  every program answers all 110 questions (generator only)
  2. judge   the judge model labels answers that need it: did it decline,
             and yes/no for answers that don't start with "Yes," / "No,"
  3. ragas   (--ragas) RAGAS faithfulness + answer relevancy on the baseline's
             suite answers, to compare failing and passing answers
  4. report  per-test results and failure rates -> summary.md, using the
             reviewed labels in label_review.csv where they exist

Usage:
    python -m src.checklist --ragas          # everything
    python -m src.checklist --report         # only rebuild the report
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime

import ollama
import pandas as pd

from . import config
from .evaluate import abstained, answer_questions, ragas_scores
from .rag import RAG, configure_lm

SUITE_PATH = config.EVAL_DIR / "checklist.json"
OUT_DIR = config.RESULTS_DIR / "checklist"
# Every distinct label the judge produced (did it decline; yes/no when the answer
# doesn't open with "Yes,"/"No,"), re-read by a reviewer. The report uses the
# reviewed labels and states how often the judge agreed.
REVIEW_PATH = OUT_DIR / "label_review.csv"
PROGRAMS = ["baseline"] + [f"{opt}_seed{s}" for opt in ("bootstrap", "mipro") for s in config.SEEDS]

YES_NO_PROMPT = """A QUESTION was asked and an ANSWER was given.

QUESTION: {question}
ANSWER: {answer}

Read as a reply to the QUESTION, does the ANSWER mean yes or no?
Reply with exactly one word: YES, NO, or UNCLEAR."""


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_suite() -> dict:
    return json.loads(SUITE_PATH.read_text())


def suite_questions(suite: dict) -> list[dict]:
    """Every question the programs answer, with a stable id like inv03/typo."""
    items = []
    for t in suite["invariance"]:
        items.append({"id": f"{t['id']}/original", "type": "invariance", "question": t["original"]})
        for v in t["variants"]:
            items.append({"id": f"{t['id']}/{v['kind']}", "type": "invariance", "question": v["question"]})
    for t in suite["directional"]:
        items.append({"id": f"{t['id']}/original", "type": "directional", "question": t["original"]})
        items.append({"id": f"{t['id']}/negated", "type": "directional", "question": t["negated"]})
    for t in suite["abstention"]:
        items.append({"id": t["id"], "type": "abstention", "question": t["question"]})
    return [{**it, "ground_truth": ""} for it in items]


def has_facts(answer: str, expect: list[list[str]]) -> bool:
    """Every group must match at least one of its patterns (case-insensitive)."""
    return all(any(re.search(p, answer, re.I) for p in group) for group in expect)


def leading_yes_no(answer: str) -> str | None:
    """'Yes, ...' / 'No.' -> yes / no. 'No engineer may ...' is NOT a leading no."""
    m = re.match(r"\s*(yes|no)\s*([,.!;:]|$)", answer, re.I)
    return m.group(1).lower() if m else None


def judge_yes_no(question: str, answer: str, client: ollama.Client) -> str:
    reply = client.chat(model=config.JUDGE_MODEL, options={"temperature": 0}, messages=[
        {"role": "user", "content": YES_NO_PROMPT.format(question=question, answer=answer)}])
    word = reply["message"]["content"].strip().upper()
    return "yes" if word.startswith("YES") else "no" if word.startswith("NO") else "unclear"


def program(name: str) -> RAG:
    rag = RAG()
    if name != "baseline":
        rag.load(str(config.PROGRAMS_DIR / f"{name}.json"))
    return rag


def phase_answer(programs: list[str], items: list[dict]) -> None:
    configure_lm(cache=False)
    for name in programs:
        path = OUT_DIR / f"answers_{name}.csv"
        if path.exists():
            continue
        log(f"answering: {name}")
        rows = answer_questions(program(name), items)
        df = pd.DataFrame(rows)
        df["contexts"] = df["contexts"].map(lambda cs: "\n---\n".join(cs))
        df.to_csv(path, index=False)


def phase_judge(programs: list[str], suite: dict) -> None:
    """Judge labels: declined (abstention + failed invariance) and yes/no (directional)."""
    client = ollama.Client(host=config.OLLAMA_BASE_URL)
    expect = {t["id"]: t["expect"] for t in suite["invariance"]}
    for name in programs:
        path = OUT_DIR / f"judged_{name}.csv"
        if path.exists():
            continue
        log(f"judging: {name}")
        df = pd.read_csv(OUT_DIR / f"answers_{name}.csv", keep_default_na=False)
        yes_no, yes_no_by, declined = [], [], []
        for r in df.itertuples():
            yn, by, dec = "", "", float("nan")
            if r.type == "directional":
                yn = leading_yes_no(r.answer)
                by = "parse" if yn else "judge"
                yn = yn or judge_yes_no(r.question, r.answer, client)
            if r.type == "abstention" or (r.type == "invariance" and not has_facts(r.answer, expect[r.id.split("/")[0]])):
                dec = abstained(r.question, r.answer, client)   # why a fact check failed: declined or wrong
            yes_no.append(yn); yes_no_by.append(by); declined.append(dec)
        df["yes_no"], df["yes_no_by"], df["declined"] = yes_no, yes_no_by, declined
        df.to_csv(path, index=False)


def phase_ragas() -> None:
    path = OUT_DIR / "ragas_baseline.csv"
    if path.exists():
        return
    log("RAGAS on the baseline's suite answers")
    df = pd.read_csv(OUT_DIR / "answers_baseline.csv", keep_default_na=False)
    rows = [{"id": r.id, "question": r.question, "answer": r.answer, "ground_truth": "",
             "contexts": r.contexts.split("\n---\n")} for r in df.itertuples()]
    ragas_scores(rows, ["faithfulness", "answer_relevancy"]).to_csv(path, index=False)


def test_results(name: str, suite: dict) -> pd.DataFrame:
    """One row per test: type, kind, passed, and the answers involved."""
    df = pd.read_csv(OUT_DIR / f"judged_{name}.csv", keep_default_na=False)
    df["declined"] = pd.to_numeric(df["declined"], errors="coerce")
    if REVIEW_PATH.exists():
        review = pd.read_csv(REVIEW_PATH, keep_default_na=False)
        for field in ("declined", "yes_no"):
            r = review[review.field == field].set_index(["id", "answer"])["reviewed"]
            keys = list(zip(df["id"], df["answer"]))
            hit = [k in r.index for k in keys]
            values = [r[k] for k, h in zip(keys, hit) if h]
            df.loc[hit, field] = pd.to_numeric(values) if field == "declined" else values
    df = df.set_index("id")
    rows = []
    for t in suite["invariance"]:
        orig = df.loc[f"{t['id']}/original"]
        orig_ok = has_facts(orig.answer, t["expect"])
        for v in t["variants"]:
            var = df.loc[f"{t['id']}/{v['kind']}"]
            var_ok = has_facts(var.answer, t["expect"])
            cause = "" if var_ok == orig_ok else ("declined" if (var.declined if not var_ok else orig.declined) == 1 else "wrong fact")
            rows.append({"test": f"{t['id']}/{v['kind']}", "type": "invariance", "kind": v["kind"],
                         "passed": var_ok == orig_ok, "original_correct": orig_ok, "variant_correct": var_ok,
                         "cause": cause, "answer_ids": f"{t['id']}/original|{t['id']}/{v['kind']}"})
    for t in suite["directional"]:
        o, n = df.loc[f"{t['id']}/original"], df.loc[f"{t['id']}/negated"]
        rows.append({"test": t["id"], "type": "directional", "kind": t["negation"],
                     "passed": n.yes_no == t["negated_expect"], "original_correct": o.yes_no == t["expect"],
                     "variant_correct": n.yes_no == t["negated_expect"], "no_flip": o.yes_no == n.yes_no,
                     "cause": "" if n.yes_no == t["negated_expect"] else f"answered {n.yes_no}",
                     "answer_ids": f"{t['id']}/original|{t['id']}/negated"})
    for t in suite["abstention"]:
        dec = df.loc[t["id"]].declined == 1
        rows.append({"test": t["id"], "type": "abstention", "kind": t["kind"], "passed": dec,
                     "cause": "" if dec else "answered", "answer_ids": t["id"]})
    return pd.DataFrame(rows).assign(program=name)


def report(programs: list[str], suite: dict) -> None:
    done = [p for p in programs if (OUT_DIR / f"judged_{p}.csv").exists()]
    if "baseline" not in done:
        log("baseline not judged yet")
        return
    tests = pd.concat([test_results(p, suite) for p in done])
    tests["config"] = tests["program"].str.replace(r"_seed\d+", "", regex=True)
    tests.to_csv(OUT_DIR / "test_results.csv", index=False)

    def rates(group_cols):
        per_prog = (1 - tests.groupby(["config", "program", *group_cols])["passed"].mean()).rename("fail")
        agg = per_prog.groupby(["config", *group_cols]).agg(["mean", "std", "count"])
        return agg

    order = [c for c in ["baseline", "bootstrap", "mipro"] if c in set(tests["config"])]
    by_type = rates(["type"])
    by_kind = rates(["type", "kind"])

    def cell(agg, cfg, key):
        row = agg.loc[(cfg, *key)]
        return f"{row['mean']:.0%}" if cfg == "baseline" else f"{row['mean']:.0%} ± {row['std']:.0%}"

    n_tests = tests[tests.program == "baseline"].groupby(["type", "kind"]).size()
    lines = ["| Test | Tests | " + " | ".join(order) + " |", "|---|---|" + "---|" * len(order)]
    for typ in ["invariance", "directional", "abstention"]:
        n = int(n_tests.loc[typ].sum())
        lines.append(f"| **{typ}** | {n} | " + " | ".join(f"**{cell(by_type, c, (typ,))}**" for c in order) + " |")
        for kind in n_tests.loc[typ].index:
            lines.append(f"|   {kind} | {int(n_tests.loc[(typ, kind)])} | "
                         + " | ".join(cell(by_kind, c, (typ, kind)) for c in order) + " |")

    # Context for the directional and invariance failures.
    base = tests[tests.program == "baseline"]
    d = base[base.type == "directional"]
    i = base[base.type == "invariance"]
    lines += ["", f"Baseline directional: originals correct {int(d.original_correct.sum())}/{len(d)}, "
              f"negations correct {int(d.variant_correct.sum())}/{len(d)}, same yes/no for both {int(d.no_flip.sum())}/{len(d)}.",
              f"Baseline invariance: originals correct {int(i.original_correct.sum() / 3)}/15 groups; "
              f"failure causes {i[~i.passed].cause.value_counts().to_dict()}."]

    if REVIEW_PATH.exists():
        rv = pd.read_csv(REVIEW_PATH, keep_default_na=False)
        rv["judge"], rv["reviewed"] = rv["judge"].astype(str), rv["reviewed"].astype(str)
        for field, label in [("declined", "declined or not"), ("yes_no", "yes/no")]:
            sub = rv[rv.field == field]
            lines.append(f"Judge vs review, {label}: agreed on {(sub.judge == sub.reviewed).mean():.0%} of {len(sub)} distinct answers.")

    ragas_path = OUT_DIR / "ragas_baseline.csv"
    if ragas_path.exists():
        rg = pd.read_csv(ragas_path).set_index("id")
        failed_ids = set()
        for r in base[~base.passed].itertuples():
            ids = r.answer_ids.split("|")
            if r.type == "invariance":   # the side whose fact check failed
                ids = [ids[1]] if r.original_correct else [ids[0]]
            elif r.type == "directional":
                ids = [ids[1]]
            failed_ids.update(ids)
        rg["failed"] = rg.index.isin(failed_ids)
        lines += ["", "| Baseline answers (RAGAS) | Answers | Faithfulness | Answer relevancy |", "|---|---|---|---|"]
        for label, sub in [("all suite answers", rg), ("answers that passed", rg[~rg.failed]),
                           ("answers that failed a test", rg[rg.failed])]:
            lines.append(f"| {label} | {len(sub)} | {sub.faithfulness.mean():.3f} ({sub.faithfulness.notna().sum()} scored) "
                         f"| {sub.answer_relevancy.mean():.3f} |")
        rg.to_csv(OUT_DIR / "ragas_baseline_labeled.csv")

    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--programs", nargs="*", default=PROGRAMS)
    parser.add_argument("--ragas", action="store_true", help="also score the baseline's answers with RAGAS")
    parser.add_argument("--report", action="store_true", help="only rebuild the report")
    args = parser.parse_args()

    suite = load_suite()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.report:
        items = suite_questions(suite)
        log(f"{len(items)} questions x {len(args.programs)} programs")
        phase_answer(args.programs, items)
        phase_judge(args.programs, suite)
        if args.ragas:
            phase_ragas()
    report(args.programs, suite)


if __name__ == "__main__":
    main()
