# dspy-rag: full evaluation

The complete write-up behind the [README](README.md): design choices, every result table, the
CheckList behavioral tests, the roadmap and the limitations. Raw answers, judge labels and the
reviewed grades are in `eval/results/`.

## Why this design

| Choice | Reason |
|---|---|
| **DSPy** for the pipeline | Declare the task as a typed `Signature`; an optimizer *compiles* the prompt from data instead of hand-tuning strings. |
| **RAGAS** for evaluation | Turns "looks good" into numbers: faithfulness, answer relevancy, context precision/recall. |
| **Separate generator & judge models** | Answering and *grading* are different jobs. The generator (`llama3.2:3b`) is optimized for fluent answers; the judge (`qwen2.5:7b`) is chosen for disciplined, parseable JSON verdicts and is larger than the generator, so it is not grading its own level of mistakes. |
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
EVALUATION               judge (qwen2.5:7b) ──▶ 4 RAGAS metrics + abstention check
                                    │
OPTIMIZATION             DSPy BootstrapFewShot / MIPROv2 ──▶ better prompt (compiled from the train split)
```

## Results

**Setup.** 58 gold questions over the corpus: 36 single-hop, 12 multi-hop (facts
from two sections or documents, some needing a small inference) and 10
unanswerable (the corpus doesn't say). They were split once, stratified by type,
into **16 train** questions (optimizer only) and **42 held-out test** questions
(every number below). Each configuration ran **5 times** (seeds 1–5) with the DSPy
cache off. Generator `llama3.2:3b` at temperature 0; judge `qwen2.5:7b`.
BootstrapFewShot and MIPROv2 (`auto="light"`, the 3B generator also proposing
instructions) were recompiled for every seed, so their spread includes which
demos and instructions the seed led them to pick. The baseline has no
optimizer, so its spread is judge noise only, which turned out to be close to
zero. The full experiment took about 10 hours on an M3 MacBook Air.

| Metric (35 answerable test questions unless noted) | Baseline | BootstrapFewShot | MIPROv2 |
|---|---|---|---|
| **Correctness, reviewed** (1 / 0.5 / 0 per answer) | 0.829 ± 0.000 | 0.871 ± 0.042 | **0.880 ± 0.039** |
|   single-hop (26) / multi-hop (9) | 0.865 / 0.722 | 0.942 / 0.667 | 0.954 / 0.667 |
| Correctness, 7B judge | 0.757 ± 0.000 | 0.863 ± 0.013 | 0.806 ± 0.024 |
| Gold-answer similarity (what the optimizers maximize) | 0.756 ± 0.000 | 0.829 ± 0.057 | 0.833 ± 0.046 |
| Faithfulness (RAGAS) | 0.957 ± 0.000 | 0.931 ± 0.019 | 0.949 ± 0.023 |
| Answer relevancy (RAGAS) | 0.660 ± 0.003 | 0.705 ± 0.084 | 0.725 ± 0.050 |
| Context precision / recall (RAGAS, judged once) | 0.914 / 0.971 | same retrieval | same retrieval |
| Abstention accuracy (7 unanswerable) | 0.857 ± 0.000 | 0.886 ± 0.064 | 0.886 ± 0.156 |
| Wrongly declined answerable questions | 0.000 | 0.017 | 0.023 |
| Mean answer length (words) | 15.3 | 18.2 | 19.9 |

Mean ± standard deviation over the 5 seeds. Paired bootstrap over test questions,
each configuration minus the baseline (per-question scores averaged over seeds,
10,000 resamples):

| Δ vs baseline | BootstrapFewShot | MIPROv2 |
|---|---|---|
| Correctness, reviewed | +0.043 [−0.043, +0.131] | +0.051 [−0.031, +0.137] |
| Gold-answer similarity | +0.072 [+0.015, +0.136] | +0.077 [+0.023, +0.135] |
| Faithfulness | −0.026 [−0.084, +0.034] | −0.009 [−0.051, +0.041] |
| Answer relevancy | +0.045 [−0.038, +0.133] | +0.064 [−0.009, +0.145] |

**How correctness was measured.** RAGAS doesn't check whether an answer is
*right*, so the judge also compares each answer with the gold answer, in two
yes/no checks: does it contradict it, and is it complete. To test that judge,
all 244 distinct answers to answerable test questions were graded again with
the same rubric, in a review done with Claude (Claude Code). The grades are in
`eval/results/experiment/correctness_review.csv`. The 7B judge agreed with the
review on **72%** of answers: 81% of single-hop answers but only **58%** of
multi-hop ones. When it disagreed, it was usually harsher than the review (46
answers against 22). It scored correct-but-incomplete answers as 0 instead of
0.5, and on multi-hop reasoning it erred in both directions. One example: it
gave 1 to an answer saying 15:00 "falls outside the allowed time window of
10:00-16:00", which is false, and 0 to answers that cited the real rule, the
Friday 14:00 freeze. The reviewed
column is the headline. With the judge alone, BootstrapFewShot's gain would
have looked more than twice as large (+0.106).

### The result in one paragraph

On 42 held-out questions, five seeds each, both optimizers clearly improved the
metric they optimize, similarity to the gold answers (+0.07 to +0.08, 95% CIs
above zero). But reviewed correctness rose only from 0.83 to 0.87 (BootstrapFewShot)
and 0.88 (MIPROv2), and those gains are within question-sampling uncertainty
(95% CIs of about ±0.09 that include zero). The direction is consistent: 4 of 5
BootstrapFewShot seeds and all 5 MIPROv2 seeds beat the baseline. The gain is
all on single-hop questions, where the few-shot demos make answers more complete
(asked for the travel meal allowance, the baseline answers "$75"; the optimized
programs answer "$75 in North America and €70 in Europe").
On the 9 multi-hop questions, both optimizers did slightly worse than the
baseline (0.72 to 0.67), with large seed-to-seed swings. The optimizers mostly
changed answer *style*, making answers longer (15 to 18–20 words). Similarity
and RAGAS relevancy reward that style even when correctness does not change.
BootstrapFewShot's seed 4 had the lowest similarity of any run (0.73) and tied
for the best correctness (0.90). The seeds also showed failure modes that a
single run would have hidden. Three of the five MIPROv2 seeds picked
instructions without the "say you don't know" rule, and in two of them
abstention fell from 6/7 to 5/7. With a 3B proposer, 4 of 5 MIPROv2 seeds
chose narrow, topic-specific instructions (one says to answer "using ONLY the
provided context that mentions flight booking policies"). The 7B judge's correctness grades agreed
with the review on only 58% of multi-hop answers. Retrieval was not the
bottleneck (context recall 0.97). The one retrieval miss is a multi-hop question
whose password-rotation chunk never reaches the top 4. The honest conclusion:
on this corpus, prompt optimization makes a 3B model's answers more complete on
simple lookups. It does not yet improve multi-step reasoning, and at this test
size its benefit is suggestive, not statistically established.

### Earlier single-run result (superseded)

The first version of this README reported a single run on 12 questions with a
3B judge: faithfulness 0.77 → 0.68 and relevancy 0.72 → 0.81 after
BootstrapFewShot. That evaluation also scored the 8 questions the optimizer had
trained on. With a held-out split, five seeds and a 7B judge, the direction of
both moves repeats (faithfulness −0.03, relevancy +0.05), but neither is
statistically distinguishable from zero.

## Behavioral tests (CheckList)

An average like "faithfulness 0.96" says how good typical answers are, not how
the system breaks. Following CheckList (Ribeiro et al., ACL 2020),
`eval/checklist.json` holds 110 targeted questions in three test types.
`src/checklist.py` runs them against the baseline and all 10 optimized programs
from the experiment (BootstrapFewShot and MIPROv2, 5 seeds each).

| Test type | What changes | What must happen | Tests | How it's checked |
|---|---|---|---|---|
| Invariance | a factual question rephrased 3 ways: synonyms, restructured, casual with typos | the key fact stays the same | 45 (15 × 3) | keyword rules (e.g. `\b20\b` for "20 days"), no judge |
| Directional | a yes/no question negated ("allowed" → "prohibited", "required" → "not required") | the yes/no flips | 16 pairs | literal "Yes," / "No," opening; otherwise the judge, reviewed |
| Abstention | questions the documents don't answer: 12 near-misses next to a covered fact, 6 off-topic | the answer makes no unsupported claim | 18 | the judge, every label reviewed |

Failure rates (share of tests failed; optimized = mean ± std over 5 seeds):

| Test | Tests | Baseline | BootstrapFewShot | MIPROv2 |
|---|---|---|---|---|
| **Invariance (paraphrases)** | 45 | **9%** | **2% ± 3%** | **6% ± 8%** |
|   synonym / restructure / typo | 15 each | 13% / 7% / 7% | 4% / 1% / 1% | 9% / 4% / 5% |
| **Directional (negations)** | 16 | **31%** | **12% ± 0%** | **18% ± 14%** |
|   antonym / "not" | 10 / 6 | 30% / 33% | 12% / 13% | 16% / 20% |
| **Abstention** | 18 | **22%** | **22% ± 10%** | **23% ± 5%** |
|   near-miss / off-topic | 12 / 6 | 25% / 17% | 27% / 13% | 33% / 3% |

**How the labels were checked.** I read all 60 baseline answers to the
invariance questions: the keyword rules failed nothing by mistake. One pass is
debatable: an answer that says "the context does not specify" and then gives
the 10 minutes anyway. Every label the judge produced was re-read in a review
done with Claude (148 distinct abstention answers, 28 yes/no readings). The
judge agreed on 89% and 86% of them. All 16 of its abstention errors went the
same way: it counted a correct refusal as an answer, for example "…is not
specified for low-severity incidents". So the judge alone would have overstated
abstention failures. The reviewed labels are in
`eval/results/checklist/label_review.csv`.

### What the RAGAS averages hid

- **Negation is the weakest behavior.** The baseline answers 15 of 16 yes/no
  questions correctly but only 11 of the 16 negated versions. Its five failures
  fall into three kinds:
  - *Wrong conclusions (2).* "Yes, an engineer is barred from deploying to
    production at 11:00 on a Tuesday because it occurs outside the designated
    daylight deploy hours." 11:00 is inside the window.
  - *Self-contradictions (2).* Asked "Is economy *not* the standard class for
    flights under 6 hours?", it answers "Yes, economy class is the standard…".
    The explanation is right and the yes/no is backwards.
  - *Refusal (1).* One negated question got "I don't know".
- **Abstention failures are near-misses, and they are scope transfers.** The
  model applies a real rule to a neighboring case the documents don't cover:
  - AES-256 "for Confidential data", when the rule is for Restricted data;
  - a 5-minute auto-lock for phones, when the rule is for laptops;
  - the $1,000 home-office budget in the second year, when it is first-year only;
  - 8 weeks' notice for unpaid leave, which the optimized programs gave and
    the baseline didn't; that rule is for parental leave.

  The main experiment reported 86–89% abstention accuracy, but its unanswerable
  questions were mostly off-topic (a CEO, a 401(k) match). On near-misses, the
  failure rate is 25–33%.
- **Optimization fixed fragility, not hallucination.** BootstrapFewShot cut
  paraphrase failures from 9% to 2% and negation failures from 31% to 12%.
  But near-miss abstention failures didn't fall (25% → 27%), and MIPROv2's rose
  to 33%. MIPROv2 also introduced false refusals of answerable questions: 12 of
  its 14 paraphrase failures are refusals. Several use a phrase picked up from
  its demonstrations, "Not supplied for this particular example."
- **RAGAS scored the failures as passes, or the refusals as failures.** I ran
  RAGAS faithfulness and answer relevancy on the baseline's 110 suite answers:

  | Baseline suite answers | Answers | Faithfulness | Answer relevancy |
  |---|---|---|---|
  | answerable questions, test passed | 85 | 0.982 | 0.69 |
  | answerable questions, test failed | 7 | 0.476 | 0.54 |
  | unanswerable, correctly declined | 14 | 0.25 | 0.11 |
  | unanswerable, hallucinated an answer | 4 | 0.25 | 0.69 |

  Faithfulness gave a perfect 1.00 to 4 of the 11 failing answers: the 11:00
  "barred" answer, both self-contradictions, and the phone auto-lock. Each of
  those statements reads as supported by the retrieved text. On unanswerable
  questions, RAGAS cannot tell a correct "I don't know" from a hallucination
  (faithfulness 0.25 for both), and answer relevancy ranks the hallucinations
  higher (0.69 against 0.11). An average built from these metrics rewards a
  confident wrong answer over an honest refusal.

### The CheckList result in one paragraph

The averages describe a reliable system: faithfulness 0.96, context recall
0.97, abstention 86–89%. Behavioral tests on the same pipeline show where it
breaks. The 3B model answers 15 of 16 yes/no policy questions correctly but
fails 31% of their negations. In 2 of those 5 failures the explanation is
right and the leading "Yes"/"No" is backwards. When a question sits next to a
covered fact (phones next to laptops, Confidential next to Restricted data),
the model transfers the neighboring rule instead of declining, in 25% of
near-miss questions. Prompt optimization repaired most of the surface
fragility: BootstrapFewShot cut paraphrase failures from 9% to 2% and negation
failures from 31% to 12%, with almost no seed-to-seed spread. It did nothing
for these hallucinations: near-miss abstention failures didn't fall (25% →
27%) and rose to 33% under MIPROv2, which also learned to refuse answerable questions. RAGAS
could not have shown this. It gave a perfect faithfulness score to 4 of the 11
failing baseline answers, scored correct refusals no better than
hallucinations (0.25 faithfulness each), and gave the hallucinations higher
answer relevancy (0.69 against 0.11). Average metrics measure how good typical
answers are. Targeted tests are needed to find out how the system fails, and
the LLM judge used for the labels needed checking too: it agreed with a review
on 86–89% of them.

## What I would do next (roadmap)

Done: multi-seed runs with mean ± std and paired CIs, MIPROv2, a 58-question
set with multi-hop and unanswerable questions, a 7B judge, a check of the
judge against a full review, and CheckList behavioral tests.

1. **Bigger test set, especially multi-hop.** With 35 answerable questions, the
   95% CI on a difference is about ±0.09. Detecting a 0.05 gain needs roughly
   4× the questions, and 9 multi-hop questions are too few to conclude anything.
2. **Optimize for correctness, not similarity.** The optimizers' metric is
   embedding similarity, which rewards gold-like phrasing. A completeness and
   contradiction metric, or the judge itself, would point the search at the
   right thing.
3. **A stronger instruction proposer for MIPROv2.** The 3B model proposed
   narrow, topic-specific instructions, and in 3 of 5 seeds dropped the
   abstention rule. Use a larger proposer, or pin that rule in the signature so it can't
   be optimized away.
4. **Improve retrieval for multi-hop.** Heading-aware chunking, a reranker, or
   k > 4 for questions that span documents (the 90-day question misses its
   password chunk).
5. **Target the failures CheckList found.** Add negated and near-miss
   questions to the optimizer's training split, and pin "decline when the
   documents don't cover the exact case" in the signature. Then re-run the
   suite to see whether near-miss abstention moves.
6. **A stronger correctness judge.** The 7B judge is fine for RAGAS-style
   checks but unreliable on multi-step answers (58% agreement). Use a larger
   local model or a frontier API model as the judge, or as a second opinion
   on disagreements.

## Project layout

```
data/                    # the knowledge base (fictional company policy .md files)
eval/qa_gold.json        # 58 gold questions: type, train/test split, ground truth
eval/checklist.json      # 110 CheckList tests: paraphrases, negation pairs, unanswerable
eval/results/experiment/ # per-run CSVs, per_run_summary.csv, paired_deltas.csv, summary.md
eval/results/checklist/  # answers, judge labels, label_review.csv, test_results.csv, summary.md
models/experiment/       # the compiled program for every optimizer x seed
src/config.py            # all tunable settings (models, chunking, top-k, seeds)
src/ingest.py            # chunk + embed + store into ChromaDB
src/retriever.py         # ChromaDB-backed semantic search
src/rag.py               # DSPy Signature + RAG module (retrieve → generate)
src/gold.py              # gold-set loading (split, answerable)
src/evaluate.py          # RAGAS + abstention scoring with a separate judge model
src/optimize.py          # BootstrapFewShot / MIPROv2 prompt optimization (seeded)
src/experiment.py        # multi-seed experiment runner and statistics
src/checklist.py         # CheckList behavioral tests and report
src/ask.py               # interactive CLI
```

## Honest limitations

- **The CheckList suite is small and hand-written**: 15 paraphrase groups, 16
  negation pairs and 18 unanswerable questions. One failure moves a rate by
  6–7 points, and 3 of the baseline's 4 paraphrase failures come from a single
  original question it refused ("What language is the robot fleet firmware
  written in?").

- **The corpus is fictional and small** by design, so ground-truth answers are
  known and the local model can't have memorized them — every correct answer
  must come from retrieval. The flip side: 42 test questions over 4 short
  documents is a small test, and the confidence intervals above say so.
- **The headline correctness grades and the CheckList judge labels come from a
  review done with Claude**, not from independent human annotators. The rubric is fixed (1 / 0.5 / 0), and
  every grade is published next to the 7B judge's grade so anyone can re-check
  them.
- **RAGAS metrics are LLM estimates.** Answer relevancy in particular scored
  short correct answers ("Within 12 months.", "Concord portal") near 0, and
  scored any "I don't know" as 0. That is why it is not the headline metric
  and why abstention is measured separately.
- **Retrieval metrics were judged once**, because retrieval is identical across
  seeds and configurations (the runner verifies this). Their numbers carry no
  run-to-run spread by construction.
