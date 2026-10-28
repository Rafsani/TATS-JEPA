#!/bin/bash
#SBATCH -N 1            # number of nodes
#SBATCH -c 8          # number of cores
#SBATCH --mem=64G       # amount of memory
#SBATCH -G a100:1       # GPU
#SBATCH -t 0-04:00:00     # time in d-hh:mm:ss
#SBATCH -p general      # partition
#SBATCH -q grp_twu02       # QOS
#SBATCH -o sbatch_scripts/logs/slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e sbatch_scripts/logs/slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment

# How the REINFORCE / PPO / GRPO masking changes over pretraining, from the
# policy snapshots of the pre-train-snap.sh runs. Can be run while those runs
# are still going: it uses whatever snapshots exist so far.
#
# Usage (from the jepa/ directory):
#   sbatch visualize-mask-evolution.sh
#
# Writes mask-evolution/{evolution.png,heatmaps_clipXX.png,stats_over_epochs.json}
# and mask-evolution.log

module load mamba/latest

source activate /scratch/frafsani/V-Jepa/vjepa_env

# -- matplotlib is installed outside the training environment
export PYTHONPATH=/scratch/frafsani/V-Jepa/viz_pydeps

python visualize_mask_evolution.py --out_dir mask-evolution "$@" > mask-evolution.log 2>&1
echo "finished with exit code $?"
tail -n 5 mask-evolution.log
