#import deepspeedPatch
import json
import random
from comet_ml import Experiment
from trl import OnlineDPOConfig, OnlineDPOTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from ExpertJudge import WIMJudge
import ExpertChat
from accelerate import Accelerator
import torch
import deepspeed

system_prompt = ("You should answer the question to the best of your abilities and only output the answer. " + 
                "If the question looks like a completion task, please output the completion only.")

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

# This adds a system prompt to all of the prompts
def add_system_prompt(example):
    for item in example["prompt"]:
        if item["role"] == "user":
            item["content"] = f"{system_prompt}{item['content']}"
    return example

# Get the config.json info
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Initialize Comet.ml experiment
experiment = None
num_digits = 19
experi_num = random.randint(10**(num_digits-1), 10**num_digits - 1)
experiment = Experiment(
    api_key=config.get("comet_api_key"),
    project_name=config.get("project_name"),
    workspace=config.get("workspace"),
    experiment_key="wimTestingResults"+str(experi_num)
        )

# Model getting trained. Init empty weights for a device map
llama_path = ExpertChat.get_working_dir() + '/Models/Meta-Llama-3-8B-Instruct'
with deepspeed.zero.Init():
    model = AutoModelForCausalLM.from_pretrained(llama_path, torch_dtype=torch.float32)

# Preventing the ref_model from being created a second time
with deepspeed.zero.Init():
    ref_model = AutoModelForCausalLM.from_pretrained(llama_path, torch_dtype=torch.float16)
    wrapped_ref_model = NoMoveModelWrapper(ref_model)

# Using the model's tokenizer. Setting the padding token if needed
tokenizer = AutoTokenizer.from_pretrained(llama_path, padding=True, return_tensors="pt")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Standard dataset for prompts. Including system prompt.
dataset_path = ExpertChat.get_working_dir() + '/Datasets/ultrafeedback-prompt'
train_dataset = load_dataset(dataset_path, split="train")
train_dataset = train_dataset.map(add_system_prompt)

# Custom judge for the WIM method
zeta_val = 0.4
judge = WIMJudge(model_name='llama', zeta=zeta_val)

# Adding the logger
metric_logger = MetricLoggerCallback(experiment)

# Adjust parameters for different results
model_output_dir = '/home/nstrang2/scratch/Meta-Llama-3-8B-Instruct-OnlineDPO-WIM-Zeta' + str(zeta_val)
training_args = OnlineDPOConfig(
    output_dir=model_output_dir, 
    logging_steps=10,
    save_total_limit=3,
    save_steps=50,
    save_strategy="steps",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    fp16=False,                # DeepSpeed will handle this
    bf16=False,
    deepspeed="ds_config.json",
)

trainer = OnlineDPOTrainer(
    model=model, 
    ref_model=wrapped_ref_model,
    judge=judge, 
    args=training_args, 
    processing_class=tokenizer, 
    train_dataset=train_dataset,
    callbacks=[metric_logger]
)

print("Starting training")
trainer.train()