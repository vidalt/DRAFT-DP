#!/bin/bash
#SBATCH --array=0-104
#SBATCH --time=05:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=112G  
#SBATCH -o slurm_out/slurmout_%A_%a.out
#SBATCH -e slurm_out/slurmout_%A_%a.errarray


python run_experiments_scalability.py --expe_id=$SLURM_ARRAY_TASK_ID