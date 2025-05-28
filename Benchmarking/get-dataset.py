from datasets import load_dataset, get_dataset_config_names
from huggingface_hub import snapshot_download
import os
import json

# If using a protected dataset
from huggingface_hub import login

# Get the config.json info
with open('hf_token.json', 'r') as config_file:
    config = json.load(config_file)
login(token=config.get('token'))

# env
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/lustre07/scratch/nstrang2/HFCache"
os.environ["HF_DATASETS_CACHE"] = "/lustre07/scratch/nstrang2/HFCache"
CACHE_DIR = "/lustre07/scratch/nstrang2/HFCache"

# config
name = 'Idavidrein/gpqa'
task_names = get_dataset_config_names(name)

snapshot_download(
    repo_id=name,
    repo_type="dataset",
    cache_dir=CACHE_DIR,
    local_dir_use_symlinks=False,
)

# all tasks
all_datasets = {}
for task in task_names:
    try:
        builder = load_dataset(name, name=task, split=None)
        available_splits = builder.keys()

        preferred_split = "test" if "test" in available_splits else "train"

        dataset = load_dataset(name, name=task, split=preferred_split, cache_dir=CACHE_DIR)
        print(f"Loaded {task} with split '{preferred_split}' ({len(dataset)} samples).")

    except Exception as e:
        print(f"Failed to load task '{task}': {e}")