from trl import OnlineDPOConfig, OnlineDPOTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from ExpertJudge import WIMJudge
import ExpertChat

llama_path = ExpertChat.get_working_dir() + '/Models/Meta-Llama-3-8B-Instruct'

# Model getting trained
model = AutoModelForCausalLM.from_pretrained(llama_path)
tokenizer = AutoTokenizer.from_pretrained(llama_path)

# Standard dataset for prompts
train_dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train")

# Custom judge for the WIM method
judge = WIMJudge(model_name='mixtral', zeta=0.4)

# Adjust parameters for different results
training_args = OnlineDPOConfig(output_dir='/home/nstrang2/scratch/Meta-Llama-3-8B-Instruct-OnlineDPO-WIM', 
    logging_steps=10,
    save_total_limit=3,
    save_steps=50,
    evaluation_strategy="steps",
    save_strategy="steps"
)

trainer = OnlineDPOTrainer(
    model=model, judge=judge, args=training_args, processing_class=tokenizer, train_dataset=train_dataset
)
trainer.train()