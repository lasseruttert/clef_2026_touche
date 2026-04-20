"""Char-span <-> token-level BIO conversion using HuggingFace fast tokenizer offset_mapping."""
from typing import List, Tuple

from .config import LABEL2ID


def spans_to_bio(offsets: List[Tuple[int, int]], spans: List[Tuple[int, int]],
                 special_mask: List[int]) -> List[int]:
    """Produce BIO label ids aligned with tokens given char-level entity spans.

    offsets: (start, end) per token from tokenizer(..., return_offsets_mapping=True)
    special_mask: 1 if token is special/padding (label = -100), 0 otherwise.
    """
    labels = [LABEL2ID["O"]] * len(offsets)
    for start, end in spans:
        first = True
        for i, (ts, te) in enumerate(offsets):
            if special_mask[i]:
                continue
            if ts == te:  # empty
                continue
            # token overlaps with span
            if ts >= start and te <= end:
                labels[i] = LABEL2ID["B-ENT"] if first else LABEL2ID["I-ENT"]
                first = False
    for i, m in enumerate(special_mask):
        if m:
            labels[i] = -100
    return labels


def bio_to_spans(offsets: List[Tuple[int, int]], label_ids: List[int]) -> List[Tuple[int, int]]:
    """Decode char-level spans from predicted BIO labels."""
    spans = []
    cur_start = None
    cur_end = None
    for (ts, te), lab in zip(offsets, label_ids):
        if lab == -100 or ts == te:
            continue
        name = {v: k for k, v in LABEL2ID.items()}[lab]
        if name == "B-ENT":
            if cur_start is not None:
                spans.append((cur_start, cur_end))
            cur_start, cur_end = ts, te
        elif name == "I-ENT" and cur_start is not None:
            cur_end = te
        else:  # O
            if cur_start is not None:
                spans.append((cur_start, cur_end))
                cur_start, cur_end = None, None
    if cur_start is not None:
        spans.append((cur_start, cur_end))
    return spans
