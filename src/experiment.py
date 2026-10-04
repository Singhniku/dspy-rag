"""Run every configuration once per seed on the held-out test split; report mean ± std.

Configurations:
  baseline   the uncompiled RAG program
  bootstrap  BootstrapFewShot, recompiled per seed (train order shuffled by seed)
  mipro      MIPROv2 (auto="light"), recompiled per seed         [--mipro]

Two kinds of uncertainty are reported:
  - run-to-run: mean ± std over the seeds. For the baseline this is judge noise
    plus any generator nondeterminism; for an optimizer it also includes which
    demos/instructions the seed led it to pick.
  - question sampling: a paired bootstrap over test questions (each config vs
    baseline, per-question scores averaged over seeds) gives a 95% CI for the
    difference. With 35 answerable test questions this is usually the wider one.

Retrieval is deterministic and identical across seeds and configurations (same
retriever, same questions; the optimizers only change the generation prompt),
so context precision/recall are judged once, on the first baseline run, and
copied to the others; the runner checks every run really saw the same contexts.
Judging them again would only re-measure a near-zero-temperature judge call.

Order: every seed of baseline and bootstrap first, then MIPROv2, so the core
comparison is complete even if the optional optimizer is stopped early.

Resumable: each (config, seed) result is saved under eval/results/experiment/
and skipped on a re-run; compiled programs are saved under models/experiment/.

Usage:
    python -m src.experiment                 # baseline + bootstrap, 5 seeds
    python -m src.experiment --mipro         # also MIPROv2
    python -m src.experiment --report        # only rebuild the summary
    python -m src.experiment --report --backfill   # add correctness to older runs first
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

from . import config
import ollama

from .evaluate import GENERATION, RETRIEVAL, answer_questions, correctness, score_rows, summarize
from .gold import load_gold
from .optimize import compile_program
from .rag import RAG, configure_lm

CONFIGS = ["baseline", "bootstrap", "mipro"]
PAIRED = ["correctness_reviewed", "correctness", "faithfulness", "answer_relevancy", "gold_similarity", "abstained"]
# Every distinct answer to an answerable test question, graded by a reviewer with
# the same rubric as the correctness judge (1 / 0.5 / 0). Used to check the judge.
REVIEW_PATH = config.EXPERIMENT_DIR / "correctness_review.csv"


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def run_path(cfg: str, seed: int):
    return config.EXPERIMENT_DIR / f"{cfg}_seed{seed}.csv"


def build_program(cfg: str, seed: int) -> tuple[RAG, float]:
    if cfg == "baseline":
        return RAG(), 0.0
    path = config.PROGRAMS_DIR / f"{cfg}_seed{seed}.json"
    rag = RAG()
    if path.exists():
        rag.load(str(path))
        return rag, 0.0
    t = time.time()
    rag = compile_program(cfg, seed)
    config.PROGRAMS_DIR.mkdir(parents=True, exist_ok=True)
    rag.save(str(path))
    return rag, time.time() - t


def run_one(cfg: str, seed: int, items: list[dict]) -> None:
    rag, t_compile = build_program(cfg, seed)
    t = time.time()
    rows = answer_questions(rag, items)
    t_answer = time.time() - t

    t = time.time()
    first = run_path("baseline", config.SEEDS[0])
    judge_retrieval = cfg == "baseline" and seed == config.SEEDS[0]
    df = score_rows(rows, with_retrieval=judge_retrieval)
    if not judge_retrieval:
        base = pd.read_csv(first)
        same = base.set_index("id")["contexts"].reindex(df["id"]).tolist() == df["contexts"].tolist()
        if not same:
            log(f"WARNING {cfg} seed {seed}: retrieved contexts differ from the first baseline run")
        df = df.merge(base[["id", *RETRIEVAL]], on="id", how="left")
    t_judge = time.time() - t

    config.EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(run_path(cfg, seed), index=False)
    meta = {"config": cfg, "seed": seed, "compile_s": round(t_compile), "answer_s": round(t_answer),
            "judge_s": round(t_judge), "generator": config.OLLAMA_MODEL, "judge": config.JUDGE_MODEL}
    with open(config.EXPERIMENT_DIR / "runs.jsonl", "a") as f:
        f.write(json.dumps(meta) + "\n")
    s = summarize(df)
    log(f"done {cfg} seed {seed}: faith {s['faithfulness']:.3f} relev {s['answer_relevancy']:.3f} "
        f"sim {s['gold_similarity']:.3f} abstain-acc {s['abstention_accuracy']:.2f} "
        f"(compile {t_compile:.0f}s, answer {t_answer:.0f}s, judge {t_judge:.0f}s)")


def backfill_correctness() -> None:
    """Add the correctness column to saved runs scored before it existed."""
    client = ollama.Client(host=config.OLLAMA_BASE_URL)
    for path in sorted(config.EXPERIMENT_DIR.glob("*_seed*.csv")):
        df = pd.read_csv(path)
        if "correctness" in df:
            continue
        df["correctness"] = [
            correctness(r.question, r.ground_truth, r.answer if isinstance(r.answer, str) else "", client)
            if r.type != "unanswerable" else float("nan")
            for r in df.itertuples()
        ]
        df.to_csv(path, index=False)
        log(f"backfilled correctness: {path.name} ({df['correctness'].mean():.3f})")


def paired_bootstrap(a: pd.Series, b: pd.Series, n: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (b - a) over questions with a 95% bootstrap CI; a and b share an index."""
    d = (b - a).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report() -> None:
    runs = {}
    for cfg in CONFIGS:
        frames = [pd.read_csv(run_path(cfg, s)).assign(seed=s) for s in config.SEEDS if run_path(cfg, s).exists()]
        if frames:
            runs[cfg] = pd.concat(frames)
    if "baseline" not in runs:
        log("no baseline runs yet")
        return
    if REVIEW_PATH.exists():
        review = pd.read_csv(REVIEW_PATH, keep_default_na=False)[["id", "answer", "reviewed"]]
        for cfg, df in runs.items():
            df["answer"] = df["answer"].fillna("")
            merged = df.merge(review, on=["id", "answer"], how="left")
            missing = merged["reviewed"].isna() & (merged["type"] != "unanswerable")
            if missing.any():
                log(f"{cfg}: {int(missing.sum())} answers not in the review file")
            runs[cfg] = merged.rename(columns={"reviewed": "correctness_reviewed"})

    # Run-to-run: summarize each seed, then mean ± std (ddof=1) across seeds.
    per_run = pd.DataFrame([
        {"config": cfg, "seed": seed, **summarize(df)}
        for cfg, all_df in runs.items() for seed, df in all_df.groupby("seed")
    ])
    metrics = [c for c in per_run.columns if c not in ("config", "seed")]
    agg = per_run.groupby("config", sort=False)[metrics].agg(["mean", "std", "count"])

    # Question sampling: paired bootstrap on per-question scores averaged over seeds.
    def per_question(df, col, types):
        sub = df[df["type"].isin(types)]
        return sub.groupby("id")[col].mean()

    deltas = []
    for cfg in [c for c in runs if c != "baseline"]:
        for col in PAIRED:
            if col not in runs[cfg]:
                continue
            types = ["unanswerable"] if col == "abstained" else ["single_hop", "multi_hop"]
            a = per_question(runs["baseline"], col, types)
            b = per_question(runs[cfg], col, types)
            mean, lo, hi = paired_bootstrap(a, b.reindex(a.index))
            deltas.append({"config": cfg, "metric": "abstention_accuracy" if col == "abstained" else col,
                           "delta": mean, "ci_low": lo, "ci_high": hi, "n_questions": int(a.notna().sum())})

    config.EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(config.EXPERIMENT_DIR / "per_run_summary.csv", index=False)
    pd.DataFrame(deltas).to_csv(config.EXPERIMENT_DIR / "paired_deltas.csv", index=False)

    lines = ["| Metric | " + " | ".join(f"{c} (n={int(agg.loc[c, (metrics[0], 'count')])})" for c in agg.index) + " |",
             "|---|" + "---|" * len(agg.index)]
    for m in metrics:
        if m == "ragas_nan":
            continue
        if m in RETRIEVAL:   # judged once and shared by every run (see module docstring)
            cells = [f"{agg.loc[c, (m, 'mean')]:.3f} (judged once)" for c in agg.index]
        else:
            cells = [f"{agg.loc[c, (m, 'mean')]:.3f} ± {agg.loc[c, (m, 'std')]:.3f}" for c in agg.index]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", "| Config vs baseline | Metric | Δ (mean over questions) | 95% CI | Questions |", "|---|---|---|---|---|"]
    for d in deltas:
        lines.append(f"| {d['config']} | {d['metric']} | {d['delta']:+.3f} | [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] | {d['n_questions']} |")
    (config.EXPERIMENT_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mipro", action="store_true", help="also run MIPROv2")
    parser.add_argument("--report", action="store_true", help="only rebuild the summary")
    parser.add_argument("--backfill", action="store_true", help="add correctness to runs saved before it existed")
    parser.add_argument("--seeds", type=int, nargs="*", default=config.SEEDS)
    args = parser.parse_args()

    if not args.report:
        configure_lm(cache=False)
        items = load_gold(split="test")
        log(f"{len(items)} test questions · mipro {args.mipro} · seeds {args.seeds} · "
            f"generator {config.OLLAMA_MODEL} · judge {config.JUDGE_MODEL}")
        phases = [["baseline", "bootstrap"]] + ([["mipro"]] if args.mipro else [])
        for phase in phases:
            for seed in args.seeds:
                for cfg in phase:
                    if run_path(cfg, seed).exists():
                        log(f"skip {cfg} seed {seed} (already saved)")
                        continue
                    log(f"start {cfg} seed {seed}")
                    run_one(cfg, seed, items)
            report()
    else:
        if args.backfill:
            backfill_correctness()
        report()


if __name__ == "__main__":
    main()
