import logging
from datetime import datetime
from pathlib import Path

import transformers
from transformers import TrainerCallback
from transformers.trainer_callback import PrinterCallback, ProgressCallback

_logger = logging.getLogger(__name__)


def quiet_transformers():
    transformers.logging.set_verbosity_error()
    logging.getLogger("transformers").setLevel(logging.ERROR)


def setup_logging(run_dir: Path, tag: str):
    run_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    _logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    _logger.addHandler(ch)
    fh = logging.FileHandler(run_dir / f"{tag}.log")
    fh.setFormatter(fmt)
    _logger.addHandler(fh)


class PrettyLog(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        _logger.info(
            f"[train] epoch {logs.get('epoch', 0):.2f}  "
            f"step {state.global_step:>5}  "
            f"loss {logs['loss']:.4f}  "
            f"lr {logs.get('learning_rate', 0):.2e}"
        )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        parts = [f"{k.replace('eval_', '')}={v:.4f}"
                 for k, v in metrics.items()
                 if k.startswith("eval_") and isinstance(v, float)
                 and "runtime" not in k and "per_second" not in k]
        _logger.info(f"[eval ] epoch {metrics.get('epoch', 0):.2f}  " + "  ".join(parts))


def log_best_result(results_log: Path, tag: str, metrics: dict):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    with open(results_log, "a") as f:
        f.write(f"{ts}  {tag:<30}  {parts}\n")


def attach(trainer):
    """Install PrettyLog and silence Trainer's default dict-dumping callbacks."""
    for cb in (PrinterCallback, ProgressCallback):
        trainer.remove_callback(cb)
    trainer.add_callback(PrettyLog())
