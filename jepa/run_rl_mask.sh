#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${1:-$ROOT_DIR/configs/pretrain/vitl16.yaml}"
shift || true
DEVICE="${VJEPA_DEVICE:-cuda:0}"

LOG_DIR="$ROOT_DIR/Jepa-logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/rl_mask_$(date +%Y%m%d_%H%M%S).log"
TMP_CONFIG="$LOG_DIR/.rl_mask_$(date +%Y%m%d_%H%M%S).yaml"
cleanup() {
  rm -f "$TMP_CONFIG"
}
trap cleanup EXIT

python - "$CONFIG_FILE" "$TMP_CONFIG" <<'PY'
import sys
from pathlib import Path

import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
with src.open('r') as f:
    cfg = yaml.safe_load(f)

rl_mask = cfg.setdefault('rl_mask', {})
rl_mask['enabled'] = True
rl_mask.setdefault('embed_dim', 256)
rl_mask.setdefault('depth', 2)
rl_mask.setdefault('num_heads', 4)
rl_mask.setdefault('mlp_ratio', 4.0)
rl_mask.setdefault('temperature', 1.0)
rl_mask.setdefault('entropy_coef', 0.01)
rl_mask.setdefault('baseline_momentum', 0.9)
rl_mask.setdefault('reward_scale', 1.0)
rl_mask.setdefault('lr', cfg.get('optimization', {}).get('start_lr', 2e-4))
rl_mask.setdefault('weight_decay', cfg.get('optimization', {}).get('weight_decay', 0.04))

with dst.open('w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

cd "$ROOT_DIR"
python -m app.main --fname "$TMP_CONFIG" --devices "$DEVICE" "$@" 2>&1 | tee -a "$LOG_FILE"

echo "Log written to: $LOG_FILE"
