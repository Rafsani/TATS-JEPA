#!/bin/bash
#SBATCH -N 1            # number of nodes
#SBATCH -c 8          # number of cores
#SBATCH --mem=80G       # amount of memory
#SBATCH -G a100:1       # GPU
#SBATCH -t 5-00:00:00     # time in d-hh:mm:ss
#SBATCH -p general      # partition
#SBATCH -q grp_twu02       # QOS
#SBATCH -o sbatch_scripts/logs/slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e sbatch_scripts/logs/slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment

# Pretraining with mask-policy snapshots: runs the normal trainer and, next
# to it, snapshot_policy.py, which keeps a copy of the mask policy every few
# epochs in <logging.folder>/policy-snapshots/.
#
# Usage (from the jepa/ directory):
#   sbatch pre-train-snap.sh configs/pretrain/vitl16_traj_snap.yaml pretrain_vitl16_traj_snap.log
#   sbatch pre-train-snap.sh configs/pretrain/vitl16_ppo_snap.yaml  pretrain_vitl16_ppo_snap.log
#   sbatch pre-train-snap.sh configs/pretrain/vitl16_grpo_snap.yaml pretrain_vitl16_grpo_snap.log

module load mamba/latest

source activate /scratch/frafsani/V-Jepa/vjepa_env

CONFIG=$1
LOG=$2
FOLDER=$(python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['logging']['folder'])" "$CONFIG")
TAG=$(python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['logging']['write_tag'])" "$CONFIG")
mkdir -p "$FOLDER"

python -m app.main --fname "$CONFIG" --devices cuda:0 > "$LOG" 2>&1 &
TRAIN_PID=$!

# -- restart the watcher if it ever dies; it exits 0 once training has ended
( until python snapshot_policy.py --log_dir "$FOLDER" --tag "$TAG" --pid "$TRAIN_PID" >> "$FOLDER/snapshot.log" 2>&1; do
    sleep 30
  done ) &
WATCH_PID=$!

wait "$TRAIN_PID"
STATUS=$?
wait "$WATCH_PID"
echo "training exited with code $STATUS"
ls "$FOLDER/policy-snapshots"
exit $STATUS
