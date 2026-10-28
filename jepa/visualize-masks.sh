#!/bin/bash
#SBATCH -N 1            # number of nodes
#SBATCH -c 8          # number of cores
#SBATCH --mem=64G       # amount of memory
#SBATCH -G a100:1       # GPU
#SBATCH -t 0-02:00:00     # time in d-hh:mm:ss
#SBATCH -p general      # partition
#SBATCH -q grp_twu02       # QOS
#SBATCH -o sbatch_scripts/logs/slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e sbatch_scripts/logs/slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment

# Where do the REINFORCE / PPO / GRPO mask policies mask?
# Runs visualize_masks.py on train, val and test clips.
#
# Usage (from the jepa/ directory):
#   sbatch visualize-masks.sh                 # default runs (reinforce, ppo, grpo)
#   sbatch visualize-masks.sh --runs ppo=Jepa-logs-ppo/jepa-ppo-latest.pth.tar
#
# Writes mask-viz/<split>/{clip_XX.png,summary.png,stats.json} and mask-viz/<split>.log

module load mamba/latest

source activate /scratch/frafsani/V-Jepa/vjepa_env

# -- matplotlib is installed outside the training environment
export PYTHONPATH=/scratch/frafsani/V-Jepa/viz_pydeps

mkdir -p mask-viz

for split in train val test; do
    python visualize_masks.py --split "$split" --with_init --out_dir "mask-viz/$split" "$@" > "mask-viz/$split.log" 2>&1
    echo "$split finished with exit code $?"
    tail -n 9 "mask-viz/$split.log"
done
