"""
How the masking of REINFORCE / PPO / GRPO changes during pretraining.

Reads the policy snapshots written by snapshot_policy.py
(<log_dir>/policy-snapshots/policy-epXXX.pt), runs every snapshot on the same
clips (see visualize_masks.py for how P(masked) is measured) and plots, per
epoch, where each policy masks.

Outputs (in --out_dir):
  evolution.png          per epoch: P(masked) of background / tissue tokens,
                         high- minus low-motion P(masked), correlation with
                         motion, how many tokens the policy really uses
  heatmaps_clipXX.png    one clip: rows = epochs, columns = runs, each cell
                         the clip with P(masked) averaged over its 16 frames
  stats_over_epochs.json the numbers behind evolution.png

PPO and GRPO use random masks for their first warmup_epochs (10): the policy
is not used, nor trained, before then (shaded in evolution.png).

Usage (from the jepa/ directory):
  python visualize_mask_evolution.py --out_dir mask-evolution
"""

import argparse
import glob
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.masks.trajectory_rl import compute_motion_energy
from visualize_masks import NUM_FRAMES, PATCH, TUBELET, load_clip, masking_frequency, pick_videos, read_run, spearman

DEFAULT_RUNS = [
    'reinforce=Jepa-logs-traj-snap',
    'ppo=Jepa-logs-ppo-snap',
    'grpo=Jepa-logs-grpo-snap',
]
COLORS = {'reinforce': '#2a9d8f', 'ppo': '#e76f51', 'grpo': '#6a4c93'}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', type=str, nargs='+', default=DEFAULT_RUNS, help='<name>=<log dir>')
    p.add_argument('--out_dir', type=str, default='mask-evolution')
    p.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    p.add_argument('--num_clips', type=int, default=32)
    p.add_argument('--num_samples', type=int, default=100, help='masks sampled per clip and snapshot')
    p.add_argument('--num_heatmap_clips', type=int, default=3)
    p.add_argument('--num_heatmap_epochs', type=int, default=10, help='epochs (rows) in the heatmap figures')
    p.add_argument('--frame_step', type=int, default=4)
    p.add_argument('--crop_size', type=int, default=224)
    p.add_argument('--bg_thresh', type=float, default=0.06)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def prepare_clips(args, device):
    grid = (NUM_FRAMES // TUBELET, args.crop_size // PATCH, args.crop_size // PATCH)
    clips = []
    for c, (path, labels) in enumerate(pick_videos(args)):
        clip, frames, title = load_clip(path, labels, args, random.Random(args.seed + c))
        clip = clip.to(device)
        motion = compute_motion_energy(clip.unsqueeze(0), grid[0], grid[1], PATCH)[0]
        gray = frames.float().mean(-1) / 255.
        intensity = F.avg_pool2d(gray.view(grid[0], TUBELET, *gray.shape[1:]).mean(1, keepdim=True),
                                 PATCH).squeeze(1).to(device)
        clips.append(dict(clip=clip, frames=frames, title=title, motion=motion, intensity=intensity))
    return clips, grid


def summarize(p, probs, clips, bg_thresh):
    """p: [C, D, h, w] P(masked); probs: [C, N] policy distribution."""
    motion = torch.stack([c['motion'] for c in clips])
    intensity = torch.stack([c['intensity'] for c in clips])
    background = intensity < bg_thresh
    tissue = ~background
    hi_m, lo_m = torch.zeros_like(tissue), torch.zeros_like(tissue)
    for c in range(p.size(0)):
        m = motion[c][tissue[c]]
        if m.numel() > 4:
            hi_m[c] = tissue[c] & (motion[c] >= m.quantile(0.75))
            lo_m[c] = tissue[c] & (motion[c] <= m.quantile(0.25))
    mean_of = lambda sel: float(p[sel].mean()) if sel.any() else float('nan')
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
    return dict(
        overall=float(p.mean()),
        background=mean_of(background),
        tissue=mean_of(tissue),
        low_motion=mean_of(lo_m),
        high_motion=mean_of(hi_m),
        corr_motion=float(np.mean([spearman(p[c].flatten(), motion[c].flatten()) for c in range(p.size(0))])),
        corr_intensity=float(np.mean([spearman(p[c].flatten(), intensity[c].flatten())
                                      for c in range(p.size(0))])),
        effective_tokens=float(entropy.exp().mean()),
        top10_mass=float(probs.topk(10, dim=-1).values.sum(-1).mean()),
    )


def warmup_epochs(log_dir):
    params_path = os.path.join(log_dir, 'policy-snapshots', 'params-pretrain.yaml')
    if not os.path.exists(params_path):
        return 0
    rl = yaml.safe_load(open(params_path)).get('rl_mask', {}) or {}
    for algo in ('ppo', 'grpo'):
        if algo in rl:
            return rl[algo].get('warmup_epochs', 10)
    return 0


def plot_evolution(path, results, warmups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    panels = [
        ('background', 'P(masked): background tokens'),
        ('tissue', 'P(masked): tissue tokens'),
        ('motion_gap', 'P(masked): high-motion minus low-motion tissue'),
        ('corr_motion', 'correlation of P(masked) with motion'),
        ('effective_tokens', 'tokens the policy really uses (exp entropy)'),
        ('top10_mass', 'probability on the 10 most likely tokens'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, title) in zip(axes.flat, panels):
        for name, rows in results.items():
            epochs = [r['epoch'] for r in rows]
            vals = [r['high_motion'] - r['low_motion'] if key == 'motion_gap' else r[key] for r in rows]
            ax.plot(epochs, vals, marker='o', ms=3, label=name, color=COLORS.get(name.split('-')[0]))
            if warmups.get(name):
                ax.axvspan(0, warmups[name], color=COLORS.get(name.split('-')[0], 'grey'), alpha=0.06)
        if key in ('motion_gap', 'corr_motion'):
            ax.axhline(0, color='k', lw=0.8)
        if key == 'effective_tokens':
            ax.set_yscale('log')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('epoch')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Mask policies during pretraining (shaded: PPO/GRPO warmup with random masks)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_heatmaps(path, clip, maps, title):
    """maps: {run name: [(epoch, P(masked) [D, h, w]), ...]}"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = list(maps)
    rows = max(len(v) for v in maps.values())
    fig, axes = plt.subplots(rows, len(names), figsize=(2.6 * len(names), 2.4 * rows), squeeze=False)
    frame = clip['frames'][NUM_FRAMES // 2].numpy()
    H = frame.shape[0]
    for j, name in enumerate(names):
        for i in range(rows):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if i >= len(maps[name]):
                ax.axis('off')
                continue
            epoch, p = maps[name][i]
            p = p.mean(0)  # average over the clip's temporal slices
            lo, hi = float(p.min()), float(p.max())
            ax.imshow(frame)
            ax.imshow(F.interpolate(p[None, None].float(), size=(H, H), mode='nearest')[0, 0].cpu().numpy(),
                      cmap='viridis', alpha=0.6, vmin=lo, vmax=max(hi, lo + 1e-6))
            ax.set_title(f'{name} ep {epoch}  ({lo:.2f}-{hi:.2f})', fontsize=8)
    fig.suptitle(f'{title}\nP(masked) averaged over the clip (yellow = masked more often)', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clips, grid = prepare_clips(args, device)

    results, warmups, maps = {}, {}, {}
    for spec in args.runs:
        name, log_dir = spec.split('=', 1)
        snaps = sorted(glob.glob(os.path.join(log_dir, 'policy-snapshots', 'policy-ep*.pt')))
        if not snaps:
            print(f'{name}: no snapshots in {log_dir}/policy-snapshots, skipped')
            continue
        warmups[name] = warmup_epochs(log_dir)
        heat_idx = set(np.linspace(0, len(snaps) - 1, min(args.num_heatmap_epochs, len(snaps))).round().astype(int))
        results[name], maps[name] = [], [[] for _ in range(min(args.num_heatmap_clips, len(clips)))]
        for s, snap in enumerate(snaps):
            run = read_run(f'{name}={snap}', device, with_init=False)[0]
            p, probs = [], []
            for c, clip in enumerate(clips):
                p_masked, _, pr = masking_frequency(run, clip['clip'], args, random.Random(args.seed + 1000 + c))
                p.append(p_masked.view(grid))
                probs.append(pr.float())
                if s in heat_idx and c < len(maps[name]):
                    maps[name][c].append((run['epoch'], p_masked.view(grid).cpu()))
            row = summarize(torch.stack(p), torch.stack(probs), clips, args.bg_thresh)
            row['epoch'] = run['epoch']
            results[name].append(row)
            print(f"{name} ep {run['epoch']:>3}: bg {row['background']:.3f} tissue {row['tissue']:.3f} "
                  f"hi-lo motion {row['high_motion'] - row['low_motion']:+.3f} "
                  f"corr motion {row['corr_motion']:+.2f} eff tokens {row['effective_tokens']:.0f}", flush=True)
            del run

    if not results:
        print('nothing to plot')
        return
    plot_evolution(os.path.join(args.out_dir, 'evolution.png'), results, warmups)
    for c in range(min(args.num_heatmap_clips, len(clips))):
        plot_heatmaps(os.path.join(args.out_dir, f'heatmaps_clip{c:02d}.png'), clips[c],
                      {name: maps[name][c] for name in maps}, clips[c]['title'])
    with open(os.path.join(args.out_dir, 'stats_over_epochs.json'), 'w') as f:
        json.dump(dict(args=vars(args), warmup_epochs=warmups, stats=results), f, indent=1)
    print(f'Saved to {args.out_dir}/')


if __name__ == '__main__':
    main()
