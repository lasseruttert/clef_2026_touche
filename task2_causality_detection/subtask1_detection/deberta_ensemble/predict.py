import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from task2_causality_detection.common.data_paths import FILES
from task2_causality_detection.common.preprocessing import clean_loose
from task2_causality_detection.subtask1_detection.deberta_ensemble import config as C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev", choices=["train", "dev"])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--tag", default=C.TAG)
    p.add_argument("--runs_dir", default=str(C.OUTPUT_DIR))
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(str(runs_dir / f"seed_{C.SEEDS[0]}" / "best"))
    models = []
    for seed in C.SEEDS:
        m = AutoModelForSequenceClassification.from_pretrained(str(runs_dir / f"seed_{seed}" / "best"))
        m.eval().to(device)
        models.append(m)

    df = pd.read_json(FILES["detection"][args.split], lines=True)
    df["text"] = df["text"].map(clean_loose)

    with args.out.open("w", encoding="utf-8") as f:
        for i in range(0, len(df), 32):
            batch = df.iloc[i:i + 32]
            enc = tok(batch["text"].tolist(), truncation=True, max_length=C.MAX_LENGTH,
                      padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                avg_logits = sum(m(**enc).logits for m in models) / len(models)
            preds = avg_logits.argmax(-1).cpu().tolist()
            for idx, label in zip(batch["index"].tolist(), preds):
                f.write(json.dumps({"id": idx, "label": int(label), "tag": args.tag}) + "\n")


if __name__ == "__main__":
    main()
