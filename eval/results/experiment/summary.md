| Metric | baseline (n=5) | bootstrap (n=5) | mipro (n=5) |
|---|---|---|---|
| correctness_reviewed | 0.829 ± 0.000 | 0.871 ± 0.042 | 0.880 ± 0.039 |
| correctness | 0.757 ± 0.000 | 0.863 ± 0.013 | 0.806 ± 0.024 |
| faithfulness | 0.957 ± 0.000 | 0.931 ± 0.019 | 0.949 ± 0.023 |
| answer_relevancy | 0.660 ± 0.003 | 0.705 ± 0.084 | 0.725 ± 0.050 |
| context_precision | 0.914 (judged once) | 0.914 (judged once) | 0.914 (judged once) |
| context_recall | 0.971 (judged once) | 0.971 (judged once) | 0.971 (judged once) |
| gold_similarity | 0.756 ± 0.000 | 0.829 ± 0.057 | 0.833 ± 0.046 |
| correctness_reviewed_multi_hop | 0.722 ± 0.000 | 0.667 ± 0.130 | 0.667 ± 0.152 |
| correctness_multi_hop | 0.556 ± 0.000 | 0.489 ± 0.099 | 0.422 ± 0.122 |
| gold_similarity_multi_hop | 0.747 ± 0.000 | 0.747 ± 0.025 | 0.747 ± 0.070 |
| answer_words | 15.257 ± 0.000 | 18.200 ± 1.999 | 19.943 ± 2.364 |
| abstention_accuracy | 0.857 ± 0.000 | 0.886 ± 0.064 | 0.886 ± 0.156 |
| false_abstention | 0.000 ± 0.000 | 0.017 ± 0.026 | 0.023 ± 0.024 |

| Config vs baseline | Metric | Δ (mean over questions) | 95% CI | Questions |
|---|---|---|---|---|
| bootstrap | correctness_reviewed | +0.043 | [-0.043, +0.131] | 35 |
| bootstrap | correctness | +0.106 | [-0.023, +0.240] | 35 |
| bootstrap | faithfulness | -0.026 | [-0.084, +0.034] | 35 |
| bootstrap | answer_relevancy | +0.045 | [-0.038, +0.133] | 35 |
| bootstrap | gold_similarity | +0.072 | [+0.015, +0.136] | 35 |
| bootstrap | abstention_accuracy | +0.029 | [+0.000, +0.086] | 7 |
| mipro | correctness_reviewed | +0.051 | [-0.031, +0.137] | 35 |
| mipro | correctness | +0.049 | [-0.077, +0.174] | 35 |
| mipro | faithfulness | -0.009 | [-0.051, +0.041] | 35 |
| mipro | answer_relevancy | +0.064 | [-0.009, +0.145] | 35 |
| mipro | gold_similarity | +0.077 | [+0.023, +0.135] | 35 |
| mipro | abstention_accuracy | +0.029 | [-0.114, +0.229] | 7 |
