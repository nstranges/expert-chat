#!/bin/bash
#SBATCH --account=def-lenck
#SBATCH --export=ALL,DISABLE_DCGM=1
#SBATCH --time=50:00:00
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-gpu=40G
#SBATCH --mail-user=nstrang2@uwo.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Wait until DCGM is disabled on the node
while [ ! -z "$(dcgmi -v | grep 'Hostengine build info:')" ]; do
  sleep 5;
done

# Load modules
module load python/3.11

# Create and activate environment
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index transformers datasets accelerate evaluate fire tqdm

# Define paths
MODEL_PATH=/home/nstrang2/scratch/Models/Meta-Llama-3-8B-Instruct
OUTPUT_DIR=/home/nstrang2/projects/def-lenck/nstrang2/Evals
DATASET=/home/nstrang2/scratch/Datasets/SWE-bench

# inference
python -m swebench.inference.run_llama \
  --dataset_path $DATASET \
  --model_name_or_path $MODEL_PATH \
  --output_dir $OUTPUT_DIR \
  --temperature 0 \
  --split test

# evaluation
python -m swebench.evaluate.run_eval \
  --prediction_path $OUTPUT_DIR/Meta-Llama-3-8B-Instruct-SWEBench.json \
  --dataset_path $DATASET \
  --split test

