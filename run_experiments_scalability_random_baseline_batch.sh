#!/bin/bash
#SBATCH --array=0-104
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=25
#SBATCH --mem=50G  
#SBATCH -o slurm_out/slurmout_%A_%a.out
#SBATCH -e slurm_out/slurmout_%A_%a.errarray


python run_experiments_scalability_random_baseline.py --expe_id=$SLURM_ARRAY_TASK_ID