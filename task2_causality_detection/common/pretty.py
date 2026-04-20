import logging

import transformers
from transformers import TrainerCallback
from transformers.trainer_callback import PrinterCallback, ProgressCallback


def quiet_transformers():
    transformers.logging.set_verbosity_error()
    logging.getLogger("transformers").setLevel(logging.ERROR)


class PrettyLog(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        print(f"[train] epoch {logs.get('epoch', 0):.2f}  "
              f"step {state.global_step:>5}  "
              f"loss {logs['loss']:.4f}  "
              f"lr {logs.get('learning_rate', 0):.2e}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        parts = [f"{k.replace('eval_', '')}={v:.4f}"
                 for k, v in metrics.items()
                 if k.startswith("eval_") and isinstance(v, float)
                 and "runtime" not in k and "per_second" not in k]
        print(f"[eval ] epoch {metrics.get('epoch', 0):.2f}  " + "  ".join(parts))


def attach(trainer):
    """Install PrettyLog and silence Trainer's default dict-dumping callbacks."""
    for cb in (PrinterCallback, ProgressCallback):
        trainer.remove_callback(cb)
    trainer.add_callback(PrettyLog())
