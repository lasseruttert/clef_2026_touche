import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from task2_causality_detection.common.data_paths import FILES
from task2_causality_detection.common.pretty import attach, log_best_result, quiet_transformers, setup_logging
from task2_causality_detection.subtask2_extraction.roberta_bio_baseline import config as C
from task2_causality_detection.subtask2_extraction.roberta_bio_baseline.bio import (
    spans_to_bio,
    bio_to_spans,
)


def load_split(split: str) -> pd.DataFrame:
    df = pd.read_json(FILES["extraction"][split], lines=True)
    # text kept verbatim (offset-preserving)
    return df[["index", "text", "entity"]]


def encode(df: pd.DataFrame, tokenizer) -> Dataset:
    enc = tokenizer(
        df["text"].tolist(),
        truncation=True,
        max_length=C.MAX_LENGTH,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    labels = []
    for i in range(len(df)):
        spans = [tuple(s) for s in df["entity"].iloc[i]]
        labels.append(spans_to_bio(enc["offset_mapping"][i], spans,
                                   enc["special_tokens_mask"][i]))
    ds = Dataset.from_dict({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
        "offset_mapping": enc["offset_mapping"],
        "index": df["index"].tolist(),
        "text": df["text"].tolist(),
        "entity": df["entity"].tolist(),
    })
    return ds


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


def main():
    quiet_transformers()
    setup_logging(C.OUTPUT_DIR, C.TAG)
    set_seed(C.SEED)
    tokenizer = AutoTokenizer.from_pretrained(C.MODEL_NAME, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(
        C.MODEL_NAME, num_labels=len(C.LABELS), id2label=C.ID2LABEL, label2id=C.LABEL2ID,
    )

    train_df = load_split("train")
    dev_df = load_split("dev")
    train_ds = encode(train_df, tokenizer)
    dev_ds = encode(dev_df, tokenizer)

    keep = ["input_ids", "attention_mask", "labels"]
    train_view = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep])
    dev_view = dev_ds.remove_columns([c for c in dev_ds.column_names if c not in keep])

    def compute_metrics(eval_pred):
        logits, _ = eval_pred
        preds = np.argmax(logits, axis=-1)
        f1s = []
        for i, row in enumerate(dev_ds):
            offsets = row["offset_mapping"]
            n = len(offsets)
            pred_ids = preds[i][:n].tolist()
            pred_ids = [-100 if row["labels"][j] == -100 else p for j, p in enumerate(pred_ids)]
            pred_spans = bio_to_spans(offsets, pred_ids)
            _, _, f1 = span_f1(pred_spans, row["entity"])
            f1s.append(f1)
        return {"span_f1": float(np.mean(f1s))}

    args = TrainingArguments(
        output_dir=str(C.OUTPUT_DIR),
        num_train_epochs=C.EPOCHS,
        per_device_train_batch_size=C.BATCH_SIZE,
        per_device_eval_batch_size=C.BATCH_SIZE * 2,
        learning_rate=C.LR,
        weight_decay=C.WEIGHT_DECAY,
        warmup_ratio=C.WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="span_f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=C.SEED,
        report_to="tensorboard",
        logging_strategy="epoch",
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_view,
        eval_dataset=dev_view,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )
    attach(trainer)
    trainer.train()
    trainer.save_model(str(C.OUTPUT_DIR / "best"))
    best = {k.replace("eval_", ""): v for k, v in trainer.evaluate().items()
            if k.startswith("eval_") and isinstance(v, float)
            and "runtime" not in k and "per_second" not in k}
    log_best_result(C.RESULTS_LOG, C.TAG, best)


if __name__ == "__main__":
    main()
