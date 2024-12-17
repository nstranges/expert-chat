#!/bin/bash
#SBATCH --account=def-lenck
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem-per-gpu=16G
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=nstrang2@uwo.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Wait until DCGM is disabled on the node
while [ ! -z "$(dcgmi -v | grep 'Hostengine build info:')" ]; do
  sleep 5;
done

# Load modules
module load python/3.11
module load gcc arrow/17.0.0

# Set distributed environment variables
export DISABLE_DCGM=1
export PYTHONNOUSERSITE=1

# Create a virtual environment using venv (not virtualenv)
python -m venv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Install dependencies
pip install --no-index --upgrade pip
pip install --no-index --ignore-installed -r requirements.txt

# Activate only on the main node                                                               
source $SLURM_TMPDIR/env/bin/activate

# Set environment variables for distributed training
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Launch the training job
srun --ntasks=4 --cpus-per-task=4 \
  accelerate launch --multi_gpu --num_processes=4 --num_machines=1 --mixed_precision=fp16 \
  /home/nstrang2/projects/def-lenck/nstrang2/Code/ODPO-Trainer.py
