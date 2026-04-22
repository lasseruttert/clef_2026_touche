import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from task2_causality_detection.common.data_paths import FILES
from task2_causality_detection.common.preprocessing import clean_loose
from task2_causality_detection.deberta_joint import config as C


class _ListDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def spans_to_bio(
    offsets: List[Tuple[int, int]],
    spans: List[Tuple[int, int]],
    special_mask: List[int],
) -> List[int]:
    labels = [C.LABEL2ID["O"]] * len(offsets)
    for start, end in spans:
        first = True
        for i, (ts, te) in enumerate(offsets):
            if special_mask[i] or ts == te:
                continue
            if ts >= start and te <= end:
                labels[i] = C.LABEL2ID["B-ENT"] if first else C.LABEL2ID["I-ENT"]
                first = False
    for i, m in enumerate(special_mask):
        if m:
            labels[i] = -100
    return labels


def bio_to_spans(
    offsets: List[Tuple[int, int]], label_ids: List[int]
) -> List[Tuple[int, int]]:
    spans = []
    cur_start = cur_end = None
    for (ts, te), lab in zip(offsets, label_ids):
        if lab == -100 or ts == te:
            continue
        name = C.ID2LABEL[lab]
        if name == "B-ENT":
            if cur_start is not None:
                spans.append((cur_start, cur_end))
            cur_start, cur_end = ts, te
        elif name == "I-ENT" and cur_start is not None:
            cur_end = te
        else:
            if cur_start is not None:
                spans.append((cur_start, cur_end))
            cur_start = cur_end = None
    if cur_start is not None:
        spans.append((cur_start, cur_end))
    return spans


def span_f1(pred_spans, gold_spans):
    p = set(map(tuple, pred_spans))
    g = set(map(tuple, gold_spans))
    tp = len(p & g)
    if not p and not g:
        return 1.0, 1.0, 1.0
    prec = tp / len(p) if p else 0.0
    rec = tp / len(g) if g else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def load_detection_records(split: str, tokenizer) -> List[dict]:
    df = pd.read_json(FILES["detection"][split], lines=True)
    df["text"] = df["text"].map(clean_loose)
    enc = tokenizer(
        df["text"].tolist(),
        truncation=True,
        max_length=C.MAX_LENGTH,
    )
    records = []
    for i, row in df.iterrows():
        records.append({
            "input_ids": enc["input_ids"][i],
            "attention_mask": enc["attention_mask"][i],
            "labels": int(row["label"]),
            "index": row["index"],
        })
    return records


def _spm_offsets(tokenizer, text: str, input_ids: list) -> List[Tuple[int, int]]:
    """Compute character offsets for a slow SentencePiece tokenizer."""
    special_ids = set(tokenizer.all_special_ids)
    offsets = []
    pos = 0
    for tid in input_ids:
        if tid in special_ids:
            offsets.append((0, 0))
            continue
        token = tokenizer.convert_ids_to_tokens(tid)
        if token.startswith('\u2581'):  # ▁ marks word boundary
            surface = token[1:]
            while pos < len(text) and text[pos] in ' \t\n\r':
                pos += 1
        else:
            surface = token
        if not surface:
            offsets.append((pos, pos))
            continue
        idx = text.find(surface, pos)
        if idx != -1:
            offsets.append((idx, idx + len(surface)))
            pos = idx + len(surface)
        else:
            offsets.append((pos, pos))
    return offsets


def load_extraction_records(split: str, tokenizer) -> List[dict]:
    df = pd.read_json(FILES["extraction"][split], lines=True)
    enc = tokenizer(
        df["text"].tolist(),
        truncation=True,
        max_length=C.MAX_LENGTH,
        return_special_tokens_mask=True,
    )
    records = []
    for i, row in df.iterrows():
        offsets = _spm_offsets(tokenizer, row["text"], enc["input_ids"][i])
        spans = [tuple(s) for s in row["entity"]]
        bio_labels = spans_to_bio(offsets, spans, enc["special_tokens_mask"][i])
        records.append({
            "input_ids": enc["input_ids"][i],
            "attention_mask": enc["attention_mask"][i],
            "labels": bio_labels,
            "offset_mapping": offsets,
            "entity": row["entity"],
            "index": row["index"],
        })
    return records


def load_identification_records(split: str, tokenizer) -> List[dict]:
    df = pd.read_json(FILES["identification"][split], lines=True)
    enc = tokenizer(
        df["text"].tolist(),
        truncation=True,
        max_length=C.MAX_LENGTH,
    )
    records = []
    for i, row in df.iterrows():
        records.append({
            "input_ids": enc["input_ids"][i],
            "attention_mask": enc["attention_mask"][i],
            "labels": int(row["label"]),
            "index": row["index"],
        })
    return records


def make_collate_cls(pad_id: int):
    def collate(batch):
        input_ids = pad_sequence(
            [torch.tensor(x["input_ids"]) for x in batch],
            batch_first=True, padding_value=pad_id,
        )
        attention_mask = pad_sequence(
            [torch.tensor(x["attention_mask"]) for x in batch],
            batch_first=True, padding_value=0,
        )
        labels = torch.tensor([x["labels"] for x in batch])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    return collate


def make_collate_bio(pad_id: int):
    def collate(batch):
        input_ids = pad_sequence(
            [torch.tensor(x["input_ids"]) for x in batch],
            batch_first=True, padding_value=pad_id,
        )
        attention_mask = pad_sequence(
            [torch.tensor(x["attention_mask"]) for x in batch],
            batch_first=True, padding_value=0,
        )
        labels = pad_sequence(
            [torch.tensor(x["labels"]) for x in batch],
            batch_first=True, padding_value=-100,
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    return collate


def make_loader(records: List[dict], collate_fn, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        _ListDataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
