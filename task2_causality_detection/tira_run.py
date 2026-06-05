import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import DebertaV2Config, DebertaV2Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task2_causality_detection.common.preprocessing import clean_loose
from task2_causality_detection.common.pretty import quiet_transformers
from task2_causality_detection.deberta_joint import config as C
from task2_causality_detection.deberta_joint.data import _spm_offsets, bio_to_spans
from task2_causality_detection.deberta_joint.model import JointDeBERTa


TASKS = {
    "st1": {
        "task_id": 0,
        "input_stem": "causality-detection",
        "output_name": "st1_predictions.jsonl",
    },
    "st2": {
        "task_id": 1,
        "input_stem": "causal-candidate-extraction",
        "output_name": "st2_predictions.jsonl",
    },
    "st3": {
        "task_id": 2,
        "input_stem": "causality-identification",
        "output_name": "st3_predictions.jsonl",
    },
}


def _jsonl_files(input_dir: Path, stem: str) -> list[Path]:
    return sorted(input_dir.rglob(f"{stem}-*.jsonl"))


def _record_id(row: pd.Series):
    return row["index"] if "index" in row else row["id"]


def _output_id(row: pd.Series):
    return _record_id(row)


def _write_jsonl(path: Path, rows: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _log(message: str):
    print(f"[tira-run] {message}", flush=True)


def _deberta_config(vocab_size: int) -> DebertaV2Config:
    return DebertaV2Config(
        vocab_size=vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        relative_attention=True,
        position_buckets=256,
        max_relative_positions=-1,
        position_biased_input=False,
        norm_rel_ebd="layer_norm",
        type_vocab_size=0,
    )


def _load_model(device: torch.device, model_dir: Path | None):
    best_dir = model_dir or C.OUTPUT_DIR / "best"
    _log(f"loading model from {best_dir}")
    required_files = [
        "model.pt",
        "spm.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    missing = [name for name in required_files if not (best_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing DeBERTa joint artifact(s) in {best_dir}: {', '.join(missing)}. "
            "The TIRA submission must include task2_causality_detection/deberta_joint/runs/best "
            "or provide it via modelDir/MODEL_DIR."
        )

    checkpoint = torch.load(best_dir / "model.pt", map_location="cpu")
    tokenizer = DebertaV2Tokenizer.from_pretrained(str(best_dir))

    model = JointDeBERTa(_deberta_config(checkpoint["vocab_size"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    _log("model loaded")
    return model, tokenizer


def _predict_cls(model, tokenizer, device: torch.device, input_file: Path, task_id: int):
    df = pd.read_json(input_file, lines=True)
    texts = df["text"].map(clean_loose).tolist() if task_id == 0 else df["text"].tolist()
    ids = [_output_id(row) for _, row in df.iterrows()]

    rows = []
    with torch.no_grad():
        for start in range(0, len(df), 32):
            batch_texts = texts[start:start + 32]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                max_length=C.MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            )
            encoded = {
                "input_ids": encoded["input_ids"].to(device),
                "attention_mask": encoded["attention_mask"].to(device),
            }
            logits = model(task_id=task_id, **encoded).logits
            preds = logits.argmax(-1).cpu().tolist()
            rows.extend(
                {"id": sample_id, "label": int(label), "tag": C.TAG}
                for sample_id, label in zip(ids[start:start + 32], preds)
            )
    return rows


def _predict_extraction(model, tokenizer, device: torch.device, input_file: Path):
    df = pd.read_json(input_file, lines=True)
    encoded = tokenizer(
        df["text"].tolist(),
        truncation=True,
        max_length=C.MAX_LENGTH,
        return_special_tokens_mask=True,
    )

    records = []
    for i, row in df.iterrows():
        records.append({
            "index": _output_id(row),
            "input_ids": encoded["input_ids"][i],
            "attention_mask": encoded["attention_mask"][i],
            "offset_mapping": _spm_offsets(tokenizer, row["text"], encoded["input_ids"][i]),
            "special_tokens_mask": encoded["special_tokens_mask"][i],
        })

    rows = []
    pad_id = tokenizer.pad_token_id
    with torch.no_grad():
        for start in range(0, len(records), 32):
            batch = records[start:start + 32]
            input_ids = pad_sequence(
                [torch.tensor(r["input_ids"]) for r in batch],
                batch_first=True,
                padding_value=pad_id,
            ).to(device)
            attention_mask = pad_sequence(
                [torch.tensor(r["attention_mask"]) for r in batch],
                batch_first=True,
                padding_value=0,
            ).to(device)
            logits = model(
                task_id=TASKS["st2"]["task_id"],
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits.cpu()
            for j, rec in enumerate(batch):
                length = len(rec["input_ids"])
                pred_ids = logits[j, :length].argmax(-1).tolist()
                pred_ids = [
                    -100 if rec["special_tokens_mask"][k] else pred_ids[k]
                    for k in range(length)
                ]
                spans = [[s, e] for s, e in bio_to_spans(rec["offset_mapping"], pred_ids)]
                rows.append({"id": rec["index"], "entity": spans, "tag": C.TAG})
    return rows


def _select_input(input_dir: Path, stem: str) -> Path | None:
    files = _jsonl_files(input_dir, stem)
    if not files:
        return None
    non_train = [p for p in files if "-train" not in p.name]
    return non_train[0] if non_train else files[0]


def _generic_input(input_dir: Path) -> Path | None:
    candidates = sorted(input_dir.rglob("*.jsonl"))
    candidates = [p for p in candidates if not p.name.startswith(".")]
    if len(candidates) == 1:
        return candidates[0]
    for name in ("inputs.jsonl", "input.jsonl"):
        matches = [p for p in candidates if p.name == name]
        if matches:
            return matches[0]
    return None


def _infer_generic_subtask(input_file: Path) -> str:
    lower_path = str(input_file).lower()
    if "task1" in lower_path:
        return "st1"
    if "task2" in lower_path:
        return "st2"
    if "task3" in lower_path:
        return "st3"

    df = pd.read_json(input_file, lines=True)
    if "text" not in df.columns:
        raise ValueError(f"Cannot infer subtask from {input_file}: no text column.")
    sample = "\n".join(df["text"].astype(str).head(20).tolist())
    if "<e0>" in sample and "<e1>" in sample:
        return "st3"
    if "entity" in df.columns:
        return "st2"
    return "st1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-directory",
        default=os.environ.get("inputDataset"),
        type=Path,
    )
    parser.add_argument(
        "--output-directory",
        default=os.environ.get("outputDir"),
        type=Path,
    )
    parser.add_argument(
        "--subtask",
        choices=["all", *TASKS.keys()],
        default=os.environ.get("SUBTASK", "all"),
    )
    parser.add_argument(
        "--model-directory",
        default=os.environ.get("modelDir") or os.environ.get("MODEL_DIR"),
        type=Path,
    )
    args = parser.parse_args()

    if args.input_directory is None or args.output_directory is None:
        raise SystemExit("Provide --input-directory/--output-directory or set inputDataset/outputDir.")

    quiet_transformers()
    _log(f"input directory: {args.input_directory}")
    _log(f"output directory: {args.output_directory}")
    generic_input = _generic_input(args.input_directory)
    selected = TASKS.keys() if args.subtask == "all" else [args.subtask]
    if generic_input is not None:
        selected = [_infer_generic_subtask(generic_input)] if args.subtask == "all" else [args.subtask]
        _log(f"generic input: {generic_input}; inferred subtask: {selected[0]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"device: {device}")
    model, tokenizer = _load_model(device, args.model_directory)

    written_outputs = []
    for name in selected:
        task = TASKS[name]
        input_file = generic_input or _select_input(args.input_directory, task["input_stem"])
        if input_file is None:
            continue

        if name == "st2":
            rows = _predict_extraction(model, tokenizer, device, input_file)
        else:
            rows = _predict_cls(model, tokenizer, device, input_file, task["task_id"])

        if generic_input is None:
            _write_jsonl(args.output_directory / task["output_name"], rows)
        written_outputs.append(rows)

    if not written_outputs:
        expected = ", ".join(t["input_stem"] + "-*.jsonl" for t in TASKS.values())
        raise FileNotFoundError(f"No supported input files found below {args.input_directory}. Expected {expected}.")
    if len(written_outputs) == 1:
        _write_jsonl(args.output_directory / "predictions.jsonl", written_outputs[0])


if __name__ == "__main__":
    main()
