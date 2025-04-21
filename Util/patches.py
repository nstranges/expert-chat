import os
import gc
from transformers import TrainerCallback, Trainer
import torch
import warnings
from transformers.trainer_utils import OPTIMIZER_NAME, SCHEDULER_NAME, reissue_pt_warnings

# This gets the latest checkpoint number and returns it
def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [
        os.path.join(checkpoint_dir, d)
        for d in os.listdir(checkpoint_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, d))
    ]
    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return checkpoints[-1]

# Preventing the ref_model being sent to the GPU from accelerate
class NoMoveModelWrapper:
    def __init__(self, model):
        self.model = model

    # Ignore .to() calls
    def to(self, *args, **kwargs):
        return self
    
    # For forward pass
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # Forward other attributes
    def __getattr__(self, attr):
        return getattr(self.model, attr)
    
# Log metrics during training
class MetricLoggerCallback(TrainerCallback):
    def __init__(self, experiment):
        self.experiment = experiment

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.experiment is not None:
            self.experiment.log_metrics(
                {
                    "train_loss": logs.get("loss", None),
                    "eval_loss": logs.get("eval_loss", None),
                    "reward_score": logs.get("reward_score", None),
                },
                step=state.global_step,
            )

# Creating a callback to clear the computation graph when the checkpoint is saved
class ClearCudaCacheCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[step {state.global_step}] Cleared CUD cache after checkpoint.")

def patched_load_optimizer(self, checkpoint):
    optimizer_path = os.path.join(checkpoint, OPTIMIZER_NAME)
    scheduler_path = os.path.join(checkpoint, SCHEDULER_NAME)

    # Safely force optimizer + scheduler to load on CPU
    print(f"Loading optimizer from CPU to avoid GPU OOM...")
    optimizer_state = torch.load(optimizer_path, map_location="cpu")
    self.optimizer.load_state_dict(optimizer_state)

    with warnings.catch_warnings(record=True) as caught_warnings:
        self.lr_scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
    reissue_pt_warnings(caught_warnings)
