"""
Visualise where the learned mask policies (REINFORCE / PPO / GRPO) mask.

For every run, the mask policy saved in its checkpoint is rebuilt exactly as
in training and run on the same set of clips. Many masks are sampled per
clip, giving each token's masking frequency P(masked):

  * reinforce (app/vjepa, Jepa-logs-traj): the policy places the V-JEPA
    multi-block target blocks (8 small + 2 large, one mask per config);
    masked = tokens inside the sampled blocks
  * ppo / grpo (TATS policy): the policy picks the 10% context tokens;
    masked = every token it did not pick

A masked token is a prediction target: hidden from the context encoder,
predicted by the predictor.

Outputs (in --out_dir):
  clip_XX.png    per clip: frames, motion energy, then per run P(masked)
                 and one sampled mask (masked tokens greyed out)
  summary.png    per run: average P(masked) over space, over time, and by
                 token type (background / tissue, low / high motion)
  stats.json     the numbers behind summary.png

With --with_init, each run also gets an "<name>-init" row: the same policy
freshly initialised, i.e. what it does at the start of RL training (for
ppo / grpo it uses the final patch embedding as its input).

Usage (from the jepa/ directory):
  python visualize_masks.py --out_dir mask-viz
  python visualize_masks.py --runs ppo=Jepa-logs-ppo/jepa-ppo-latest.pth.tar --split val --num_clips 4
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from eval_frame_cls import DATA_ROOT, MEAN, STD, read_frame_labels, read_train_list
from src.masks.tats_ppo import TATSPolicy, policy_log_probs, sample_tats_masks
from src.masks.trajectory_rl import (
    TrajectoryMaskPolicy,
    compute_motion_energy,
    sample_block_size,
    sample_trajectory_masks,
)
from src.models.utils.patch_embed import PatchEmbed3D

NUM_FRAMES = 16
TUBELET = 2
PATCH = 16
DEFAULT_RUNS = [
    'reinforce=Jepa-logs-traj/jepa-traj-latest.pth.tar',
    'ppo=Jepa-logs-ppo/jepa-ppo-latest.pth.tar',
    'grpo=Jepa-logs-grpo/jepa-grpo-latest.pth.tar',
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', type=str, nargs='+', default=DEFAULT_RUNS, help='<name>=<checkpoint>')
    p.add_argument('--out_dir', type=str, default='mask-viz')
    p.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                   help='where the clips come from')
    p.add_argument('--num_clips', type=int, default=64, help='clips (one per video) for the statistics')
    p.add_argument('--num_plot_clips', type=int, default=8, help='clips that get their own figure')
    p.add_argument('--num_samples', type=int, default=200, help='masks sampled per clip and run')
    p.add_argument('--frame_step', type=int, default=4, help='frame stride inside a clip (training: 4)')
    p.add_argument('--crop_size', type=int, default=224)
    p.add_argument('--bg_thresh', type=float, default=0.06,
                   help='tokens with mean intensity below this (0-1) count as background')
    p.add_argument('--with_init', action='store_true', help='also show each policy at initialisation')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Clips
# --------------------------------------------------------------------------- #

def pick_videos(args):
    """-> list of (video path, label text) for --num_clips random videos of the split."""
    if args.split == 'train':
        items = [(path, f'video label {label}') for path, label in
                 read_train_list(f'{DATA_ROOT}/train-20251119T060603Z-1-001/train/updated_jepa.csv')]
    else:
        split_dir = {
            'val': f'{DATA_ROOT}/val-20251119T054616Z-1-001/val',
            'test': f'{DATA_ROOT}/test-20251119T054614Z-1-001/test',
        }[args.split]
        items = [(path, labels) for path, labels in read_frame_labels(split_dir)]
    rng = random.Random(args.seed)
    return rng.sample(items, min(args.num_clips, len(items)))


def load_clip(path, labels, args, rng):
    """
    One 16-frame clip, sampled like the training loader (a random window of
    16 * frame_step frames), resized to crop_size (no random crop).
    -> clip [3, T, H, W] normalised, frames [T, H, W, 3] uint8, title
    """
    from decord import VideoReader, cpu
    vr = VideoReader(path, num_threads=1, ctx=cpu(0))
    span = NUM_FRAMES * args.frame_step
    start = rng.randint(0, max(len(vr) - span, 0))
    idx = np.linspace(start, start + span, num=NUM_FRAMES)
    idx = np.clip(idx, start, min(start + span, len(vr)) - 1).astype(np.int64)
    frames = torch.from_numpy(vr.get_batch(list(idx)).asnumpy()).permute(0, 3, 1, 2).float()
    frames = F.interpolate(frames, size=(args.crop_size, args.crop_size), mode='bilinear',
                           align_corners=False, antialias=True)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1) * 255.
    std = torch.tensor(STD).view(1, 3, 1, 1) * 255.
    clip = ((frames - mean) / std).permute(1, 0, 2, 3)  # [3, T, H, W]

    title = os.path.basename(path)
    if isinstance(labels, dict):
        clip_labels = [labels.get(int(i)) for i in idx]
        known = [l for l in clip_labels if l is not None]
        title += f'  frames {idx[0]}-{idx[-1]}, SP frames {sum(known)}/{len(known)}'
    else:
        title += f'  frames {idx[0]}-{idx[-1]}, {labels}'
    return clip, frames.permute(0, 2, 3, 1).clamp(0, 255).byte(), title


# --------------------------------------------------------------------------- #
#  Policies
# --------------------------------------------------------------------------- #

def read_run(spec, device, with_init):
    """-> list of run dicts (the trained policy, plus its initialisation if asked)."""
    name, ckpt_path = spec.split('=', 1)
    with open(os.path.join(os.path.dirname(ckpt_path), 'params-pretrain.yaml')) as f:
        params = yaml.safe_load(f)
    checkpoint = torch.load(ckpt_path, map_location='cpu', mmap=True, weights_only=False)
    state = {k.replace('module.', ''): v for k, v in checkpoint['mask_policy'].items()}
    algo = checkpoint.get('mask_policy_algo') or ('reinforce' if 'motion_scale' in state else None)
    if algo is None:
        raise ValueError(f'{ckpt_path}: cannot tell which mask policy this is')
    epoch = checkpoint.get('epoch')
    uniform_power = params['model'].get('uniform_power', False)
    grid_depth = NUM_FRAMES // TUBELET
    grid_size = params['data']['crop_size'] // PATCH
    rl_cfg = params.get('rl_mask', {}) or {}

    if algo == 'reinforce':
        def build():
            return TrajectoryMaskPolicy(
                img_size=params['data']['crop_size'], patch_size=PATCH, num_frames=NUM_FRAMES,
                tubelet_size=TUBELET, in_chans=3,
                embed_dim=rl_cfg.get('embed_dim', 256),
                depth=rl_cfg.get('depth', 2),
                num_heads=rl_cfg.get('num_heads', 4),
                mlp_ratio=rl_cfg.get('mlp_ratio', 4.0),
                qkv_bias=True,
                uniform_power=uniform_power)
        extra = dict(temperature=float(rl_cfg.get('temperature', 1.0)), mask_cfgs=params['mask'])
    else:
        cfg = rl_cfg.get(algo, {}) or {}
        enc_state = {k.replace('module.', '').replace('backbone.', ''): v
                     for k, v in checkpoint['encoder'].items() if 'patch_embed' in k}
        in_dim = enc_state['patch_embed.proj.weight'].size(0)
        patch_embed = PatchEmbed3D(patch_size=PATCH, tubelet_size=TUBELET, in_chans=3, embed_dim=in_dim)
        patch_embed.load_state_dict({k[len('patch_embed.'):]: v for k, v in enc_state.items()})
        patch_embed.to(device).eval()

        def build():
            return TATSPolicy(
                in_dim=in_dim, grid_depth=grid_depth, grid_size=grid_size,
                embed_dim=cfg.get('embed_dim', 384),
                traj_attn=cfg.get('traj_attn', True),
                num_blocks=cfg.get('num_traj_attn_blocks', 1),
                num_heads=cfg.get('num_heads', 8),
                mlp_ratio=cfg.get('mlp_ratio', 4.0),
                qkv_bias=cfg.get('qkv_bias', False),
                drop=cfg.get('drop_rate', 0.0),
                approx=cfg.get('traj_approx', 'orthoformer'),
                num_landmarks=cfg.get('num_landmarks', 128),
                softmax_temp=cfg.get('softmax_temp', 1.0),
                uniform_power=uniform_power)
        num_patches = grid_depth * grid_size * grid_size
        extra = dict(patch_embed=patch_embed,
                     num_visible=max(1, int(num_patches * (1 - cfg.get('mask_ratio', 0.9)))),
                     explore_eps=float(cfg.get('explore_eps', 0.05)))
    del checkpoint

    policy = build()
    msg = policy.load_state_dict(state)
    print(f'{name}: {algo} policy from {ckpt_path} (epoch {epoch}): {msg}')
    runs = [dict(name=name, algo=algo, epoch=epoch, policy=policy.to(device).eval(), **extra)]
    if with_init:
        torch.manual_seed(0)
        runs.append(dict(name=f'{name}-init', algo=algo, epoch=0, policy=build().to(device).eval(), **extra))
    return runs


@torch.no_grad()
def masking_frequency(run, clip, args, rng):
    """
    -> p_masked [N] (fraction of sampled masks hiding each token),
       one sampled mask [N] (bool, True = masked),
       policy probabilities [N] (block-centre or context-token distribution)
    """
    clips = clip.unsqueeze(0)
    S = args.num_samples
    if run['algo'] == 'reinforce':
        logits, _ = run['policy'](clips)
        logits = logits.float() / max(run['temperature'], 1e-6)
        N = logits.size(-1)
        grid_depth, grid_size = NUM_FRAMES // TUBELET, args.crop_size // PATCH
        counts, example = torch.zeros(N, device=clip.device), None
        for s in range(S):
            # -- block shapes vary per iteration in training; vary them per sample here
            specs = []
            for cfg in run['mask_cfgs']:
                t, h, w = sample_block_size(
                    seed=rng.randint(0, 2**31 - 1),
                    temporal_scale=cfg.get('temporal_scale'), spatial_scale=cfg.get('spatial_scale'),
                    aspect_ratio=cfg.get('aspect_ratio'),
                    duration=grid_depth, height=grid_size, width=grid_size)
                specs.append(dict(t=t, h=h, w=w, npred=cfg.get('num_blocks', 1),
                                  max_context_duration=max(1, int(grid_depth * cfg.get('max_temporal_keep', 1.0))),
                                  max_keep=None))
            _, masks_pred, _, _ = sample_trajectory_masks(logits, (grid_depth, grid_size, grid_size), specs)
            for m_idx, mp in enumerate(masks_pred):
                masked = torch.zeros(N, device=clip.device)
                masked[mp[0]] = 1.
                counts += masked / len(masks_pred)
                if example is None and m_idx == len(masks_pred) - 1:
                    example = masked.bool()  # a large-block mask: the one that hides most of the clip
        return counts / S, example, logits.softmax(-1)[0]

    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=clip.is_cuda):
        logits, _ = run['policy'](run['patch_embed'](clips))
    logits = logits.float()
    N = logits.size(-1)
    _, masks_pred = sample_tats_masks(logits.expand(S, -1), run['num_visible'], run['explore_eps'])
    masked = torch.zeros(S, N, device=clip.device)
    masked.scatter_(1, masks_pred, 1.)
    return masked.mean(0), masked[0].bool(), policy_log_probs(logits, run['explore_eps']).exp()[0]


# --------------------------------------------------------------------------- #
#  Plots
# --------------------------------------------------------------------------- #

def plot_clip(path, frames, motion, results, title):
    """frames [T, H, W, 3]; motion [D, h, w]; results: list of (label, p_masked [D,h,w], example [D,h,w])."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    D = motion.size(0)
    H = frames.size(1)
    rows = 2 + 2 * len(results)
    fig, axes = plt.subplots(rows, D, figsize=(1.7 * D, 1.75 * rows), squeeze=False)
    up = lambda x: F.interpolate(x[None, None].float(), size=(H, H), mode='nearest')[0, 0].cpu().numpy()

    for d in range(D):
        img = frames[d * TUBELET].numpy()
        axes[0, d].imshow(img)
        axes[0, d].set_title(f'frames {d * TUBELET}-{d * TUBELET + 1}', fontsize=7)
        axes[1, d].imshow(img)
        axes[1, d].imshow(up(motion[d]), cmap='magma', alpha=0.6, vmin=0, vmax=float(motion.max()))
        for r, (label, p_masked, example) in enumerate(results):
            lo, hi = float(p_masked.min()), float(p_masked.max())
            ax = axes[2 + 2 * r, d]
            ax.imshow(img)
            ax.imshow(up(p_masked[d]), cmap='viridis', alpha=0.6, vmin=lo, vmax=max(hi, lo + 1e-6))
            ax = axes[3 + 2 * r, d]
            shown = img.copy()
            shown[up(example[d]) > 0.5] = (0.25 * shown[up(example[d]) > 0.5]).astype(np.uint8) + 60
            ax.imshow(shown)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_ylabel('clip', fontsize=8)
    axes[1, 0].set_ylabel('motion\nenergy', fontsize=8)
    for r, (label, p_masked, _) in enumerate(results):
        axes[2 + 2 * r, 0].set_ylabel(
            f'{label}\nP(masked)\n{float(p_masked.min()):.2f}-{float(p_masked.max()):.2f}', fontsize=8)
        axes[3 + 2 * r, 0].set_ylabel(f'{label}\none mask\n(grey = masked)', fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_summary(path, mean_frame, stats, grid):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = list(stats)
    fig, axes = plt.subplots(len(names), 3, figsize=(12, 3.2 * len(names)), squeeze=False)
    H = mean_frame.shape[0]
    for r, name in enumerate(names):
        s = stats[name]
        spatial = torch.tensor(s['spatial_p_masked'])
        lo, hi = float(spatial.min()), float(spatial.max())
        ax = axes[r, 0]
        ax.imshow(mean_frame, cmap='gray')
        im = ax.imshow(F.interpolate(spatial[None, None], size=(H, H), mode='nearest')[0, 0].numpy(),
                       cmap='viridis', alpha=0.65, vmin=lo, vmax=max(hi, lo + 1e-6))
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f'{name}: P(masked) by position\n(mean over clips and time)', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[r, 1]
        temporal = s['temporal_p_masked']
        ax.plot(range(1, grid[0] + 1), temporal, marker='o')
        centre, half = float(np.mean(temporal)), max(0.05, (max(temporal) - min(temporal)) * 0.75)
        ax.set_ylim(max(0., centre - half), min(1., centre + half))
        ax.set_xlabel('temporal slice (2 frames each)')
        ax.set_ylabel('P(masked)')
        ax.set_title(f'{name}: P(masked) over time in the clip', fontsize=9)

        ax = axes[r, 2]
        keys = ['background', 'tissue', 'low_motion', 'high_motion']
        vals = [s['by_token_type'][k] for k in keys]
        ax.bar(['background', 'tissue', 'low\nmotion', 'high\nmotion'], vals,
               color=['#555555', '#c08040', '#4080c0', '#c04040'])
        ax.axhline(s['overall_p_masked'], color='k', ls='--', lw=1, label='overall')
        ax.set_ylim(max(0., min(vals) - 0.05), min(1., max(vals) + 0.05))
        ax.set_title(f'{name}: P(masked) by token type\n'
                     f'corr. with motion {s["corr_motion"]:+.2f}, with intensity {s["corr_intensity"]:+.2f}\n'
                     f'policy: {s["policy_effective_tokens"]:.0f} effective tokens, '
                     f'top-10 hold {s["policy_top10_mass"] * 100:.0f}% of the probability',
                     fontsize=8)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def spearman(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm() + 1e-12))


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    runs = [r for spec in args.runs for r in read_run(spec, device, args.with_init)]
    videos = pick_videos(args)
    grid = (NUM_FRAMES // TUBELET, args.crop_size // PATCH, args.crop_size // PATCH)

    per_run = {r['name']: dict(p=[], motion=[], intensity=[], probs=[]) for r in runs}
    frame_sum = None
    for c, (path, labels) in enumerate(videos):
        clip_rng = random.Random(args.seed + c)
        clip, frames, title = load_clip(path, labels, args, clip_rng)
        clip = clip.to(device)
        motion = compute_motion_energy(clip.unsqueeze(0), grid[0], grid[1], PATCH)[0]  # [D, h, w]
        # -- mean raw intensity (0-1) of each token, for background / tissue
        gray = frames.float().mean(-1) / 255.  # [T, H, W]
        intensity = F.avg_pool2d(gray.view(grid[0], TUBELET, *gray.shape[1:]).mean(1, keepdim=True),
                                 PATCH).squeeze(1).to(device)  # [D, h, w]
        frame_sum = gray.mean(0) if frame_sum is None else frame_sum + gray.mean(0)

        results = []
        for run in runs:
            p_masked, example, probs = masking_frequency(run, clip, args, random.Random(args.seed + 1000 + c))
            per_run[run['name']]['p'].append(p_masked.view(grid))
            per_run[run['name']]['probs'].append(probs)
            per_run[run['name']]['motion'].append(motion)
            per_run[run['name']]['intensity'].append(intensity)
            results.append((f"{run['name']} (ep {run['epoch']})", p_masked.view(grid), example.view(grid)))
        if c < args.num_plot_clips:
            plot_clip(os.path.join(args.out_dir, f'clip_{c:02d}.png'), frames, motion, results, title)
        print(f'[{c + 1}/{len(videos)}] {title}')

    stats = {}
    for name, d in per_run.items():
        p = torch.stack(d['p'])  # [C, D, h, w]
        motion = torch.stack(d['motion'])
        intensity = torch.stack(d['intensity'])
        background = intensity < args.bg_thresh
        tissue = ~background
        # -- high / low motion: top / bottom quartile of tissue tokens within each clip
        hi_m, lo_m = torch.zeros_like(tissue), torch.zeros_like(tissue)
        for c in range(p.size(0)):
            m = motion[c][tissue[c]]
            if m.numel() > 4:
                q1, q3 = m.quantile(0.25), m.quantile(0.75)
                hi_m[c] = tissue[c] & (motion[c] >= q3)
                lo_m[c] = tissue[c] & (motion[c] <= q1)
        mean_of = lambda sel: float(p[sel].mean()) if sel.any() else float('nan')
        # -- how concentrated the policy's own distribution is (block centres for
        # reinforce, context tokens for ppo / grpo); uniform = N effective tokens
        probs = torch.stack(d['probs']).float()
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
        stats[name] = dict(
            overall_p_masked=float(p.mean()),
            spatial_p_masked=p.mean(dim=(0, 1)).cpu().tolist(),
            temporal_p_masked=p.mean(dim=(0, 2, 3)).cpu().tolist(),
            by_token_type=dict(background=mean_of(background), tissue=mean_of(tissue),
                               low_motion=mean_of(lo_m), high_motion=mean_of(hi_m)),
            corr_motion=float(np.mean([spearman(p[c].flatten(), motion[c].flatten()) for c in range(p.size(0))])),
            corr_intensity=float(np.mean([spearman(p[c].flatten(), intensity[c].flatten())
                                          for c in range(p.size(0))])),
            # -- how different the masks are between clips: 0 = same mask for every clip
            between_clip_std=float(p.std(0).mean()),
            background_token_frac=float(background.float().mean()),
            policy_effective_tokens=float(entropy.exp().mean()),
            policy_top10_mass=float(probs.topk(10, dim=-1).values.sum(-1).mean()),
        )

    mean_frame = (frame_sum / len(videos)).numpy()
    plot_summary(os.path.join(args.out_dir, 'summary.png'), mean_frame, stats, grid)
    with open(os.path.join(args.out_dir, 'stats.json'), 'w') as f:
        json.dump(dict(args=vars(args), stats=stats), f, indent=1)

    print(f"\n{'run':<18}{'P(mask)':>8}{'bg':>8}{'tissue':>8}{'lowMot':>8}{'highMot':>8}"
          f"{'corrMot':>9}{'corrInt':>9}{'clipStd':>9}{'effTok':>8}{'top10':>7}")
    for name, s in stats.items():
        t = s['by_token_type']
        print(f"{name:<18}{s['overall_p_masked']:8.3f}{t['background']:8.3f}{t['tissue']:8.3f}"
              f"{t['low_motion']:8.3f}{t['high_motion']:8.3f}{s['corr_motion']:+9.2f}"
              f"{s['corr_intensity']:+9.2f}{s['between_clip_std']:9.3f}"
              f"{s['policy_effective_tokens']:8.0f}{s['policy_top10_mass']:7.2f}")
    print(f'Saved to {args.out_dir}/')


if __name__ == '__main__':
    main()
