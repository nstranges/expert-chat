#!/bin/bash
#SBATCH --export=ALL,DISABLE_DCGM=1
#SBATCH --time=168:00:00
#SBATCH --gpus=3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-gpu=80G

# Load modules
module load python/3.11
module load gcc arrow/17.0.0
module load httpproxy
module load cuda
module load mpi4py/4.0.0

# Run on each node
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

# Activate the environment
source $SLURM_TMPDIR/env/bin/activate;

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Launch the training job
accelerate launch --num_processes=1 --mixed_precision=bf16 /home/ODPO-Trainer.py
