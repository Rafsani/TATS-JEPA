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

# Frozen-encoder frame classification (standard plane vs non-standard plane)
# on the validation and test sets.
#
# Usage (from the jepa/ directory):
#   sbatch eval-frame-cls.sh <name>=<checkpoint> [<name>=<checkpoint> ...]
#   sbatch eval-frame-cls.sh random-init=none          # untrained-encoder baseline
#
# Example:
#   sbatch eval-frame-cls.sh ppo-e34=Jepa-logs-ppo-e34/jepa-ppo-e34-latest.pth.tar
#
# Each run writes eval-cls/<name>/{results.json,predictions_val.csv,predictions_test.csv}
# and its log to eval-cls/<name>.log

module load mamba/latest

source activate /scratch/frafsani/V-Jepa/vjepa_env

mkdir -p eval-cls

for spec in "$@"; do
    name="${spec%%=*}"
    ckpt="${spec#*=}"
    if [ "$ckpt" = "none" ]; then
        python eval_frame_cls.py --random_init --out_dir "eval-cls/$name" > "eval-cls/$name.log" 2>&1
    else
        python eval_frame_cls.py --checkpoint "$ckpt" --out_dir "eval-cls/$name" > "eval-cls/$name.log" 2>&1
    fi
    echo "$name finished with exit code $?"
done

# -- summary table of everything evaluated so far
python - <<'PY' > eval-cls/summary.txt
import glob, json
keys = ['acc', 'balanced_acc', 'f1', 'auroc']
print(f"{'model':<18}{'epoch':>6}  " + '  '.join(f'val_{k:<13}' for k in keys) + '  ' + '  '.join(f'test_{k:<12}' for k in keys))
for path in sorted(glob.glob('eval-cls/*/results.json')):
    r = json.load(open(path))
    print(f"{path.split('/')[1]:<18}{str(r['pretrain_epoch']):>6}  "
          + '  '.join(f"{r['val'][k]:<17.4f}" for k in keys) + '  '
          + '  '.join(f"{r['test'][k]:<17.4f}" for k in keys))
PY
cat eval-cls/summary.txt
