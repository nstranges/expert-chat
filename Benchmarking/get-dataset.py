from datasets import load_dataset, get_dataset_config_names
import os

# If using a protected dataset
# from huggingface_hub import login
# login(token="")

# env
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_DATASETS_CACHE"] = "/lustre07/scratch/nstrang2/HFCache/"

# config
name = 'google/IFEval'
task_names = get_dataset_config_names(name)

# all tasks
all_datasets = {}
for task in task_names:
    try:
        builder = load_dataset(name, name=task, split=None)
        available_splits = builder.keys()

        preferred_split = "test" if "test" in available_splits else "train"

        dataset = load_dataset(name, name=task, split=preferred_split)
        print(f"Loaded {task} with split '{preferred_split}' ({len(dataset)} samples).")

    except Exception as e:
        print(f"Failed to load task '{task}': {e}")