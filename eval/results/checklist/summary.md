| Test | Tests | baseline | bootstrap | mipro |
|---|---|---|---|---|
| **invariance** | 45 | **9%** | **2% ± 3%** | **6% ± 8%** |
|   restructure | 15 | 7% | 1% ± 3% | 4% ± 6% |
|   synonym | 15 | 13% | 4% ± 4% | 9% ± 15% |
|   typo | 15 | 7% | 1% ± 3% | 5% ± 6% |
| **directional** | 16 | **31%** | **12% ± 0%** | **18% ± 14%** |
|   antonym | 10 | 30% | 12% ± 4% | 16% ± 11% |
|   not | 6 | 33% | 13% ± 7% | 20% ± 22% |
| **abstention** | 18 | **22%** | **22% ± 10%** | **23% ± 5%** |
|   near_miss | 12 | 25% | 27% ± 14% | 33% ± 10% |
|   off_topic | 6 | 17% | 13% ± 7% | 3% ± 7% |

Baseline directional: originals correct 15/16, negations correct 11/16, same yes/no for both 4/16.
Baseline invariance: originals correct 14/15 groups; failure causes {'declined': 3, 'wrong fact': 1}.
Judge vs review, declined or not: agreed on 89% of 148 distinct answers.
Judge vs review, yes/no: agreed on 86% of 28 distinct answers.

| Baseline answers (RAGAS) | Answers | Faithfulness | Answer relevancy |
|---|---|---|---|
| all suite answers | 110 | 0.830 (110 scored) | 0.607 |
| answers that passed | 99 | 0.879 (99 scored) | 0.608 |
| answers that failed a test | 11 | 0.394 (11 scored) | 0.594 |
