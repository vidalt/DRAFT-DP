#!/bin/bash
#SBATCH --array=0-1349
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=112G  
#SBATCH -o experiments_results/output_%A_%a.out

python run_experiments.py --expe_id=$SLURM_ARRAY_TASK_ID