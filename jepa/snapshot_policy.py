"""
Keep a copy of the mask policy from a running pretraining job, epoch by epoch.

The trainers overwrite <log_dir>/<tag>-latest.pth.tar at the end of every
epoch, so only the final policy survives a run. This watcher runs next to the
training process (see pre-train-snap.sh): whenever the checkpoint has been
rewritten and has stopped changing, it loads it and saves a small file

  <out_dir>/policy-ep<epoch>.pt   {epoch, mask_policy, mask_policy_algo,
                                   encoder (patch_embed weights only)}

PPO / GRPO need the encoder's patch embedding: their policy input is
patch_embed(clip). params-pretrain.yaml is copied next to the snapshots, so
every snapshot can be read by visualize_masks.py / visualize_mask_evolution.py.

Epochs kept: every epoch up to --dense_until, then every --every epochs, plus
the last checkpoint once the training process (--pid) has exited.

Usage (from the jepa/ directory):
  python snapshot_policy.py --log_dir Jepa-logs-ppo-snap --tag jepa-ppo-snap --pid <training pid>
"""

import argparse
import os
import shutil
import time

import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--log_dir', type=str, required=True)
    p.add_argument('--tag', type=str, required=True, help='logging.write_tag of the run')
    p.add_argument('--out_dir', type=str, default=None, help='default: <log_dir>/policy-snapshots')
    p.add_argument('--pid', type=int, default=None, help='training process; stop once it has exited')
    p.add_argument('--every', type=int, default=5)
    p.add_argument('--dense_until', type=int, default=20)
    p.add_argument('--poll', type=float, default=30., help='seconds between checks')
    p.add_argument('--settle', type=float, default=60.,
                   help='the checkpoint must be unchanged for this long before it is read')
    return p.parse_args()


def alive(pid):
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def keep(epoch, args, final):
    return final or epoch <= args.dense_until or epoch % args.every == 0


def snapshot(ckpt_path, args, final):
    """Save the policy of the checkpoint at ckpt_path if its epoch is kept. -> epoch"""
    # -- a plain (not mmap) load: if the trainer starts rewriting the file
    # mid-read this raises instead of crashing the process
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    epoch = checkpoint['epoch']
    out_path = os.path.join(args.out_dir, f'policy-ep{epoch:03d}.pt')
    if checkpoint.get('mask_policy') is None:
        print(f'epoch {epoch}: checkpoint has no mask policy', flush=True)
    elif keep(epoch, args, final) and not os.path.exists(out_path):
        algo = checkpoint.get('mask_policy_algo') or (
            'reinforce' if any('motion_scale' in k for k in checkpoint['mask_policy']) else None)
        torch.save({
            'epoch': epoch,
            'mask_policy': checkpoint['mask_policy'],
            'mask_policy_algo': algo,
            'encoder': {k: v.clone() for k, v in checkpoint['encoder'].items() if 'patch_embed' in k},
        }, out_path + '.tmp')
        os.replace(out_path + '.tmp', out_path)
        print(f'epoch {epoch}: saved {out_path}', flush=True)
    del checkpoint
    return epoch


def main():
    args = parse_args()
    args.out_dir = args.out_dir or os.path.join(args.log_dir, 'policy-snapshots')
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.log_dir, f'{args.tag}-latest.pth.tar')
    params_path = os.path.join(args.log_dir, 'params-pretrain.yaml')
    print(f'watching {ckpt_path} -> {args.out_dir}', flush=True)

    seen, changed_at, done = None, None, None
    while True:
        finished = not alive(args.pid)
        if os.path.exists(params_path) and not os.path.exists(os.path.join(args.out_dir, 'params-pretrain.yaml')):
            shutil.copy(params_path, args.out_dir)
        if os.path.exists(ckpt_path):
            st = os.stat(ckpt_path)
            key = (st.st_mtime, st.st_size)
            if key != seen:
                seen, changed_at = key, time.time()
            settled = finished or time.time() - changed_at >= args.settle
            if settled and (key != done or finished):
                try:
                    snapshot(ckpt_path, args, final=finished)
                    done = key
                except Exception as e:
                    # -- most likely read while being rewritten: try again next poll
                    print(f'could not read checkpoint yet ({type(e).__name__}: {e})', flush=True)
                    seen = None
        if finished:
            print('training process has exited; stopping', flush=True)
            return
        time.sleep(args.poll)


if __name__ == '__main__':
    main()
