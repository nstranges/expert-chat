#!/bin/bash
#SBATCH --account=def-lenck
#SBATCH --job-name=zip_model
#SBATCH --output=zip_model.log
#SBATCH --time=24:00:00  # Adjust time as needed
#SBATCH --mem=512G       # Adjust memory as needed
#SBATCH --mail-user=nstrang2@uwo.ca
#SBATCH --mail-type=BEGIN,END,FAIL

cd $SLURM_TMPDIR
cp -r /home/nstrang2/scratch/Models/ $SLURM_TMPDIR/

zip -r Models.zip $SLURM_TMPDIR/Models/
cp $SLURM_TMPDIR/Models.zip /home/nstrang2/scratch/
