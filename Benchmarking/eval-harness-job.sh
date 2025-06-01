#!/bin/bash
#SBATCH --account=def-lenck
#SBATCH --export=ALL,DISABLE_DCGM=1
#SBATCH --time=20:00:00
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
module load gcc arrow/17.0.0

# Set up environment
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Eval
cd /lustre07/scratch/nstrang2/Datasets/lm-evaluation-harness/

pip install --no-index --upgrade pip
pip install /home/nstrang2/scratch/Libraries/word2number-1.1.zip
pip install --no-index langdetect
pip install --no-index immutabledict
pip install --no-index nltk
pip install --no-index packaging
pip install --no-index -e .[dev]

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_DATASETS_OFFLINE="1"
export HF_HOME="/home/nstrang2/scratch/HFCache"
export HF_DATASETS_CACHE="/home/nstrang2/scratch/HFCache"

# Run the evaluation harness. Try bbh after getting data
lm_eval \
  --model hf \
  --model_args pretrained=/home/nstrang2/scratch/Models/Meta-Llama-3-8B-Instruct \
  --tasks ifeval,bbh,gpqa,mmlu \
  --batch_size 1 \
  --output_path /home/nstrang2/projects/def-lenck/nstrang2/Evals/Meta-Llama-3-8B-Instruct.json
