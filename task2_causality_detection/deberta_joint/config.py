from pathlib import Path

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42

SPECIAL_TOKENS = ["<e0>", "</e0>", "<e1>", "</e1>"]

LABELS = ["O", "B-ENT", "I-ENT"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

OUTPUT_DIR = Path(__file__).parent / "runs"
TAG = "deberta-joint"

_BASE = Path(__file__).parents[1]
RESULTS_LOG_ST1 = _BASE / "subtask1_detection" / "results.log"
RESULTS_LOG_ST2 = _BASE / "subtask2_extraction" / "results.log"
RESULTS_LOG_ST3 = _BASE / "subtask3_identification" / "results.log"
