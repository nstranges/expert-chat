#!/bin/bash
#SBATCH --account=def-lenck
#SBATCH --export=ALL,DISABLE_DCGM=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem-per-gpu=16G
#SBATCH --mail-user=nstrang2@uwo.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Wait until DCGM is disabled on the node
while [ ! -z "$(dcgmi -v | grep 'Hostengine build info:')" ]; do
  sleep 5;
done

# Load modules
module load python/3.11
module load gcc arrow/17.0.0
module load httpproxy

# create the virtual environment on each node : 
srun --ntasks $SLURM_NNODES --tasks-per-node=1 bash << EOF
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
EOF

# activate only on main node                                                               
source $SLURM_TMPDIR/env/bin/activate;

# Launch the training job
srun --ntasks=4 accelerate launch --multi_gpu --num_processes=4 --num_machines=1 --mixed_precision=fp16 /home/nstrang2/projects/def-lenck/nstrang2/Code/ODPO-Trainer.py
