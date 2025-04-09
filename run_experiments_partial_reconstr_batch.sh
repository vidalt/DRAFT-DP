#!/bin/bash
#SBATCH --array=0-184
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=112G  
#SBATCH -o slurm_out/slurmout_%A_%a.out
#SBATCH -e slurm_out/slurmout_%A_%a.errarray


python run_experiments_partial_reconstr.py --expe_id=$SLURM_ARRAY_TASK_ID