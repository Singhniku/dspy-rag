"""The gold question set: loading and filtering.

Each item in eval/qa_gold.json has:
    id            stable identifier (q01..)
    type          single_hop | multi_hop | unanswerable
    split         train (optimizer only) | test (held out, used for every reported number)
    question, ground_truth

The split was assigned once, stratified by type with seed 42, so the optimizer
never sees a test question.
"""
from __future__ import annotations

import json

from . import config


def load_gold(split: str | None = None) -> list[dict]:
    items = json.loads(config.GOLD_QA_PATH.read_text())
    if split:
        items = [it for it in items if it["split"] == split]
    return items


def is_answerable(item: dict) -> bool:
    return item["type"] != "unanswerable"
