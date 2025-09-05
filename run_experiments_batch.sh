#!/bin/bash
#SBATCH --array=1000-1349
#SBATCH --time=03:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G  
#SBATCH -o experiments_results/output_%A_%a.out

python run_experiments.py --expe_id=$SLURM_ARRAY_TASK_ID
