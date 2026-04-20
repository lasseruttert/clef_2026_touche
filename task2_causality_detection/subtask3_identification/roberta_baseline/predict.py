import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from task2_causality_detection.common.data_paths import FILES
from task2_causality_detection.subtask3_identification.roberta_baseline import config as C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev", choices=["train", "dev"])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--tag", default=C.TAG)
    p.add_argument("--model_dir", default=str(C.OUTPUT_DIR / "best"))
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    df = pd.read_json(FILES["identification"][args.split], lines=True)

    with args.out.open("w", encoding="utf-8") as f:
        for i in range(0, len(df), 32):
            batch = df.iloc[i:i + 32]
            enc = tok(batch["text"].tolist(), truncation=True, max_length=C.MAX_LENGTH,
                      padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**enc).logits
            preds = logits.argmax(-1).cpu().tolist()
            for idx, label in zip(batch["index"].tolist(), preds):
                f.write(json.dumps({"id": idx, "label": int(label), "tag": args.tag}) + "\n")


if __name__ == "__main__":
    main()
