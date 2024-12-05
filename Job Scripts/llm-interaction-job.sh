#!/bin/bash
#SBATCH --account=def-lenck
#SBATCH --export=ALL,DISABLE_DCGM=1
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=4
#SBATCH --mem=64000M
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

# Putting the models onto the compute node
#cd $SLURM_TMPDIR
#mkdir -p $SLURM_TMPDIR/Models
# cp /home/nstrang2/scratch/Models/ $SLURM_TMPDIR/
#cp /home/nstrang2/scratch/Models.zip $SLURM_TMPDIR/
#unzip -q $SLURM_TMPDIR/Models.zip -d $SLURM_TMPDIR/
#ls -R $SLURM_TMPDIR/Models/
#echo "Model files successfully unzipped to $SLURM_TMPDIR/Models/"
#pwd > current_directory.txt
#cp ./current_directory.txt ~/projects/def-lenck/lenck/nstrang2/Code/

# srun exports the current env, which contains $VIRTUAL_ENV and $PATH variables
srun python /home/nstrang2/projects/def-lenck/nstrang2/Code/llm-interaction.py;