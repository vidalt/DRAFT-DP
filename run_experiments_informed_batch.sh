#!/bin/bash
#SBATCH --array=0-34
#SBATCH --time=35:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G  
#SBATCH -o experiments_results/output_%A_%a.out

python run_experiments_informed_attacker.py --expe_id=$SLURM_ARRAY_TASK_ID