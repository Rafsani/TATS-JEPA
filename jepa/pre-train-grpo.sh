#!/bin/bash
#SBATCH -N 1            # number of nodes
#SBATCH -c 8          # number of cores, adjust if needed to accommodate 5 parallel processes
#SBATCH --mem=80G       # amount of memory
#SBATCH -G a100:1       # GPU
#SBATCH -t 5-00:00:00     # time in d-hh:mm:ss (bumped from 2 days: RL-mask policy forward
                           # + motion-energy computation adds overhead per iteration versus
                           # the vanilla run this was cloned from -- watch the first few
                           # logged iterations' [gpu: ... ms] to see if this needs adjusting,
                           # and check your queue/QOS allows a 4-day walltime)
#SBATCH -p general      # partition
#SBATCH -q grp_twu02       # QOS
#SBATCH -o sbatch_scripts/logs/slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e sbatch_scripts/logs/slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment

module load mamba/latest

source activate /scratch/frafsani/V-Jepa/vjepa_env

python -m app.main --fname configs/pretrain/vitl16_grpo.yaml --devices cuda:0 > pretrain_vitl16_grpo.log 2>&1
