import json
import random
from comet_ml import Experiment
from trl import OnlineDPOConfig, OnlineDPOTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from ExpertJudge import WIMJudge
import ExpertChat
import torch
import os
from patches import get_latest_checkpoint, NoMoveModelWrapper, MetricLoggerCallback, ClearCudaCacheCallback, patched_load_optimizer
from peft import LoraConfig, get_peft_model

# Creating the LoRA setup for Llama
lora_parameters = {
    'r': 16,
    'lora_alpha': 16,
    'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    'lora_dropout': 0.0,
    'bias': "none",
    'task_type': "CAUSAL_LM",
}
lora_config = LoraConfig(
    r=lora_parameters['r'],
    lora_alpha=lora_parameters['lora_alpha'],
    target_modules=lora_parameters['target_modules'],
    lora_dropout=lora_parameters['lora_dropout'],
    bias=lora_parameters['bias'],
    task_type=lora_parameters['task_type'],
)

# Fix for improper loading of the optimizer
Trainer._load_optimizer_and_scheduler = patched_load_optimizer

system_prompt = ("You should answer the question to the best of your abilities and only output the answer. " + 
                "If the question looks like a completion task, please output the completion only.")

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

# Specifying the path of a potential checkpoint. Might have to load it directly from here first
zeta_val = 0.0
model_output_dir = '/home/nstrang2/scratch/Meta-Llama-3-8B-Instruct-OnlineDPO-WIM-Zeta' + str(zeta_val)
llama_path = ExpertChat.get_working_dir() + '/Models/Meta-Llama-3-8B-Instruct'

# Preventing the ref_model from being created a second time. Ref model is always loaded from the original path. Using flash attention on this too.
ref_model = AutoModelForCausalLM.from_pretrained(llama_path, device_map="auto", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_cache=False)
wrapped_ref_model = NoMoveModelWrapper(ref_model)

# Using the model's tokenizer. Setting the padding token if needed
tokenizer = AutoTokenizer.from_pretrained(llama_path, padding=True, return_tensors="pt")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load the latest checkpoint path if available
if os.path.isdir(model_output_dir) and os.listdir(model_output_dir):
     llama_path = get_latest_checkpoint(model_output_dir)
     print("Loading the checkpoint model from: " + llama_path)

# Model getting trained. Init empty weights for a device map
model = AutoModelForCausalLM.from_pretrained(llama_path, device_map="auto", attn_implementation="flash_attention_2", low_cpu_mem_usage=True, use_cache=False)

# Wrapping the model with the LoRA config
model = get_peft_model(model, lora_config)
model = model.bfloat16() # Casting the LoRA to bfloat
model.print_trainable_parameters()

# Standard dataset for prompts. Including system prompt.
dataset_path = ExpertChat.get_working_dir() + '/Datasets/ultrafeedback-prompt'
train_dataset = load_dataset(dataset_path, split="train")
train_dataset = train_dataset.map(add_system_prompt)

# Custom judge for the WIM method. Using the reference model to save memory
judge = WIMJudge(model_name='llama', zeta=zeta_val, model=wrapped_ref_model, tokenizer=tokenizer)

# Adding the logger
metric_logger = MetricLoggerCallback(experiment)

# Adjust parameters for different results
training_args = OnlineDPOConfig(
    output_dir=model_output_dir, 
    logging_steps=10,
    save_total_limit=2,
    save_steps=25,
    save_strategy="steps",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    fp16=False,                # Accelerate will handle this
    bf16=False,
    optim="paged_adamw_8bit",
)

trainer = OnlineDPOTrainer(
    model=model, 
    ref_model=wrapped_ref_model,
    judge=judge, 
    args=training_args, 
    processing_class=tokenizer, 
    train_dataset=train_dataset,
    callbacks=[metric_logger, ClearCudaCacheCallback()]
)

print("Starting training")
if os.path.isdir(model_output_dir) and os.listdir(model_output_dir):
    print("Using checkpoint")
    trainer.train(resume_from_checkpoint=True)
else:
    print("Starting fresh")
    trainer.train()

# Saving the LoRA model
model.save_pretrained(model_output_dir+"/done_model", merge_adapter=True)
tokenizer.save_pretrained(model_output_dir+"/done_model")