from pathlib import Path

from task2_causality_detection.roberta_joint.config import (
    MODEL_NAME, MAX_LENGTH, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, WARMUP_RATIO, SEED,
    SPECIAL_TOKENS, LABELS, LABEL2ID, ID2LABEL,
    RESULTS_LOG_ST1, RESULTS_LOG_ST2, RESULTS_LOG_ST3,
)

SEEDS = [42, 43, 44]
OUTPUT_DIR = Path(__file__).parent / "runs"
TAG = "roberta-joint-ensemble"
