#!/bin/bash
#SBATCH --array=0-209
#SBATCH --time=12:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G  
#SBATCH -o experiments_results/output_%A_%a.out

python run_experiments_informed_adversary_amortized_balle.py --expe_id=$SLURM_ARRAY_TASK_ID