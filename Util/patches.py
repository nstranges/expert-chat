import os
import gc
from transformers import TrainerCallback, Trainer
import torch

# Checkpoint constants
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"

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

# Loading the optimizer onto the CPU to prevent OOM errors
def patched_load_optimizer(self, checkpoint):
    if checkpoint is None:
        print("Skipping loading optimizer, starting from fresh")
        return
    
    print("Loading optimizer and scheduler on the CPU first.")
    
    # Loading the scheduler if it exists
    scheduler_path = os.path.join(checkpoint, "scheduler.pt")
    if os.path.exists(scheduler_path):
        tmp_state_dict = torch.load(scheduler_path, map_location="cpu")
        self.lr_scheduler.load_state_dict(tmp_state_dict)

    # Loading the optimizer if it exists
    optimizer_path = os.path.join(checkpoint, "optimizer.pt")
    if os.path.exists(scheduler_path):
        optimizer_state = torch.load(optimizer_path, map_location="cpu")
        self.optimizer.load_state_dict(optimizer_state)