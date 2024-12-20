import json
from comet_ml import Experiment
from trl import OnlineDPOConfig, OnlineDPOTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from ExpertJudge import WIMJudge
import ExpertChat

# Get the config.json info
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Initialize Comet.ml experiment
experiment = Experiment(
    api_key=config.get("comet_api_key"),
    project_name="online-dpo",
    workspace="your_workspace"
)

# Track system info
experiment.log_system_info()
experiment.log_system_metrics()

llama_path = ExpertChat.get_working_dir() + '/Models/Meta-Llama-3-8B-Instruct'

# Model getting trained
model = AutoModelForCausalLM.from_pretrained(llama_path)
tokenizer = AutoTokenizer.from_pretrained(llama_path)

# Standard dataset for prompts
train_dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train")

# Custom judge for the WIM method
judge = WIMJudge(model_name='mixtral', zeta=0.4)

# Adjust parameters for different results
training_args = OnlineDPOConfig(
    output_dir='/home/nstrang2/scratch/Meta-Llama-3-8B-Instruct-OnlineDPO-WIM', 
    logging_steps=10,
    save_total_limit=3,
    save_steps=50,
    evaluation_strategy="steps",
    save_strategy="steps"
)

trainer = OnlineDPOTrainer(
    model=model, 
    judge=judge, 
    args=training_args, 
    processing_class=tokenizer, 
    train_dataset=train_dataset
)

# Log metrics during training
def log_metrics(logs):
    experiment.log_metrics({
        "train_loss": logs.get('loss', None),
        "eval_loss": logs.get('eval_loss', None),
        "reward_score": logs.get('reward_score', None)
    }, step=logs.get('step', None))

trainer.add_callback(log_metrics)
trainer.train()