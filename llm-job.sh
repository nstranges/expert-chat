#!/bin/bash
#SBATCH --account=def-lenck
#SBATCH --export=ALL,DISABLE_DCGM=1
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=2
#SBATCH --mem=32000M               # memory per node
#SBATCH --mail-user=nstrang2@uwo.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Wait until DCGM is disabled on the node
while [ ! -z "$(dcgmi -v | grep 'Hostengine build info:')" ]; do
  sleep 5;
done

module load python/3.11

# create the virtual environment on each node : 
srun --ntasks $SLURM_NNODES --tasks-per-node=1 bash << EOF
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
EOF

# activate only on main node                                                               
source $SLURM_TMPDIR/env/bin/activate;

# srun exports the current env, which contains $VIRTUAL_ENV and $PATH variables
srun python /home/nstrang2/projects/def-lenck/nstrang2/Code/llm-interaction.py;