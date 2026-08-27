#!/bin/bash
#SBATCH -N 1            # number of nodes
#SBATCH -c 8          # number of cores, adjust if needed to accommodate 5 parallel processes
#SBATCH --mem=80G       # amount of memory
#SBATCH -G a100:1       # GPU
#SBATCH -t 2-00:00:00     # time in d-hh:mm:ss
#SBATCH -p general      # partition
#SBATCH -q grp_twu02       # QOS
#SBATCH -o sbatch_scripts/logs/slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e sbatch_scripts/logs/slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment

module load mamba/latest

source activate /scratch/frafsani/V-Jepa/vjepa_env

python -m app.main --fname configs/pretrain/vitl16.yaml  --devices cuda:0 > pretrain_vitl16.log 2>&1