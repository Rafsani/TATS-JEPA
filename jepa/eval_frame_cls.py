"""
Frozen-encoder frame classification probe for the IUGC ultrasound dataset.

Task: classify every frame as standard plane (SP, label 1) or non-standard
plane (NSP, label 0).

  * train: video-level labels (all frames of a video share the label)
  * val / test: per-frame labels (cls/cls_label.csv: pos_index / neg_index)

Pipeline (the pretrained V-JEPA encoder is never updated):
  1. Slide 16-frame windows over each video and run the frozen encoder.
     Its 8 temporal token slices (2 frames each) are spatially mean-pooled,
     giving one feature vector per 2 frames.
  2. Train a linear classifier on the train features.
  3. Pick the weight decay / epoch with the best *validation* score.
  4. Report that classifier on validation and test.

Usage (from the jepa/ directory):
  python eval_frame_cls.py --checkpoint Jepa-logs-ppo/jepa-ppo-latest.pth.tar --out_dir eval-cls/ppo
  python eval_frame_cls.py --random_init --out_dir eval-cls/random-init      # untrained baseline
"""

import argparse
import ast
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import src.models.vision_transformer as vit

DATA_ROOT = '/scratch/frafsani/V-Jepa/dataset/DatasetV3'
NUM_FRAMES = 16
TUBELET = 2
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=str, default=None, help='pretraining checkpoint (.pth.tar)')
    p.add_argument('--checkpoint_key', type=str, default='target_encoder',
                   help='which weights to evaluate: target_encoder (V-JEPA default) or encoder')
    p.add_argument('--random_init', action='store_true', help='evaluate an untrained encoder (baseline)')
    p.add_argument('--model_name', type=str, default='vit_large')
    p.add_argument('--out_dir', type=str, required=True)
    # -- data
    p.add_argument('--train_csv', type=str,
                   default=f'{DATA_ROOT}/train-20251119T060603Z-1-001/train/updated_jepa.csv',
                   help='"<video path><, or space><label>" per line')
    p.add_argument('--val_dir', type=str, default=f'{DATA_ROOT}/val-20251119T054616Z-1-001/val')
    p.add_argument('--test_dir', type=str, default=f'{DATA_ROOT}/test-20251119T054614Z-1-001/test')
    p.add_argument('--crop_size', type=int, default=224)
    p.add_argument('--frame_step', type=int, default=1,
                   help='temporal stride between the 16 frames of a window')
    p.add_argument('--window_stride', type=int, default=8, help='stride (in frames) between windows')
    p.add_argument('--max_train_windows', type=int, default=12, help='windows per training video')
    p.add_argument('--batch_size', type=int, default=16, help='windows per encoder forward')
    p.add_argument('--recompute_features', action='store_true', help='ignore cached features')
    # -- probe
    p.add_argument('--probe_epochs', type=int, default=30)
    p.add_argument('--probe_lr', type=float, default=1e-3)
    p.add_argument('--probe_wds', type=float, nargs='+', default=[0.0, 1e-3, 1e-2, 1e-1, 1.0])
    p.add_argument('--probe_batch_size', type=int, default=1024)
    p.add_argument('--select_metric', type=str, default='balanced_acc',
                   choices=['balanced_acc', 'acc', 'f1', 'auroc'],
                   help='validation metric used to pick the classifier')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #

def resolve_video_path(path):
    """Standard-plane train videos are stored as '<name>__<name>.avi' but listed as '<name>.avi'."""
    if os.path.exists(path):
        return path
    folder, name = os.path.split(path)
    stem, ext = os.path.splitext(name)
    doubled = os.path.join(folder, f'{stem}__{stem}{ext}')
    if os.path.exists(doubled):
        return doubled
    raise FileNotFoundError(f'video not found: {path}')


def read_train_list(path):
    """-> list of (video path, video label)."""
    items = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sep = ',' if ',' in line else ' '
            video, label = line.rsplit(sep, 1)
            items.append((resolve_video_path(video), int(label)))
    return items


def _parse_index(value, frame_count):
    value = value.strip()
    if value == 'NONE':
        return []
    if value == 'ALL':
        return list(range(frame_count))
    return [int(i) for i in ast.literal_eval(value)]


def read_frame_labels(split_dir):
    """-> list of (video path, {frame index: label})."""
    items = []
    with open(os.path.join(split_dir, 'cls', 'cls_label.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            name = row['filename'] if row['filename'].endswith('.avi') else row['filename'] + '.avi'
            frame_count = int(row['frame_count'])
            labels = {i: 1 for i in _parse_index(row['pos_index'], frame_count)}
            labels.update({i: 0 for i in _parse_index(row['neg_index'], frame_count)})
            items.append((os.path.join(split_dir, 'videos', name), labels))
    return items


def window_starts(num_video_frames, span, stride):
    """Start frames of the sliding windows; the last window is aligned to the video end."""
    last = max(num_video_frames - span, 0)
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


# --------------------------------------------------------------------------- #
#  Encoder / features
# --------------------------------------------------------------------------- #

def load_encoder(args, device):
    encoder = vit.__dict__[args.model_name](
        img_size=args.crop_size,
        patch_size=16,
        num_frames=NUM_FRAMES,
        tubelet_size=TUBELET,
        uniform_power=True,
        use_sdpa=True,
    )
    epoch = None
    if not args.random_init:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        epoch = checkpoint.get('epoch')
        state = checkpoint[args.checkpoint_key]
        state = {k.replace('module.', '').replace('backbone.', ''): v for k, v in state.items()}
        msg = encoder.load_state_dict(state, strict=True)
        print(f'Loaded {args.checkpoint_key} from {args.checkpoint} (epoch {epoch}): {msg}')
        del checkpoint
    else:
        print('Using a randomly initialised encoder (baseline)')
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder, epoch


@torch.no_grad()
def encode_windows(encoder, vr, starts, args, device):
    """
    Run the encoder on the windows starting at `starts`.
    -> [num_windows, 8, D] spatially mean-pooled feature per temporal slice.
    """
    num_video_frames = len(vr)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1, 1) * 255.
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1, 1) * 255.

    feats = []
    for b in range(0, len(starts), args.batch_size):
        batch_starts = starts[b:b + args.batch_size]
        # -- frames past the end of a short video repeat its last frame
        idx = [[min(s + i * args.frame_step, num_video_frames - 1) for i in range(NUM_FRAMES)]
               for s in batch_starts]
        flat = sorted(set(i for w in idx for i in w))
        frames = torch.from_numpy(vr.get_batch(flat).asnumpy()).to(device)  # [n, H, W, 3] uint8
        frames = frames.permute(0, 3, 1, 2).float()
        frames = F.interpolate(
            frames, size=(args.crop_size, args.crop_size), mode='bilinear', align_corners=False, antialias=True)
        pos = {f: j for j, f in enumerate(flat)}
        clips = torch.stack([frames[[pos[i] for i in w]] for w in idx])  # [B, T, 3, H, W]
        clips = (clips.permute(0, 2, 1, 3, 4) - mean) / std  # [B, 3, T, H, W]

        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            tokens = encoder(clips)  # [B, 8 * h * w, D]
        B, N, D = tokens.shape
        depth = NUM_FRAMES // TUBELET
        feats.append(tokens.float().view(B, depth, N // depth, D).mean(dim=2).cpu())
    return torch.cat(feats)


def extract_train_features(encoder, items, args, device):
    """Every temporal slice of every sampled window is one sample with the video's label."""
    import decord
    span = NUM_FRAMES * args.frame_step
    feats, labels = [], []
    for n, (path, label) in enumerate(items):
        vr = decord.VideoReader(path)
        starts = window_starts(len(vr), span, args.window_stride)
        if len(starts) > args.max_train_windows:
            keep = np.linspace(0, len(starts) - 1, args.max_train_windows).round().astype(int)
            starts = [starts[i] for i in keep]
        f = encode_windows(encoder, vr, starts, args, device).flatten(0, 1)  # [windows * 8, D]
        feats.append(f)
        labels.append(torch.full((f.size(0),), label, dtype=torch.long))
        if (n + 1) % 50 == 0 or n + 1 == len(items):
            print(f'  train: {n + 1}/{len(items)} videos', flush=True)
    return {'x': torch.cat(feats), 'y': torch.cat(labels)}


def extract_frame_features(encoder, items, args, device, split):
    """One feature per labelled frame: the slice containing it, from the window centred closest to it."""
    import decord
    span = NUM_FRAMES * args.frame_step
    feats, labels, meta = [], [], []
    for n, (path, frame_labels) in enumerate(items):
        vr = decord.VideoReader(path)
        starts = window_starts(len(vr), span, args.window_stride)
        slices = encode_windows(encoder, vr, starts, args, device)  # [W, 8, D]
        centres = np.array(starts) + (span - 1) / 2.0
        for frame, label in sorted(frame_labels.items()):
            if frame >= len(vr):
                continue
            covering = [w for w, s in enumerate(starts) if s <= frame < s + span] or [len(starts) - 1]
            w = min(covering, key=lambda i: abs(centres[i] - frame))
            t = min(max((frame - starts[w]) // (TUBELET * args.frame_step), 0), slices.size(1) - 1)
            feats.append(slices[w, t])
            labels.append(label)
            meta.append((os.path.basename(path), frame))
        if (n + 1) % 50 == 0 or n + 1 == len(items):
            print(f'  {split}: {n + 1}/{len(items)} videos', flush=True)
    return {'x': torch.stack(feats), 'y': torch.tensor(labels, dtype=torch.long), 'meta': meta}


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #

def auroc(scores, labels):
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    num_pos, num_neg = int(labels.sum()), int((1 - labels).sum())
    if num_pos == 0 or num_neg == 0:
        return float('nan')
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2.0  # average rank of each tied group
    ranks = avg_rank[inv]
    return float((ranks[labels == 1].sum() - num_pos * (num_pos + 1) / 2.0) / (num_pos * num_neg))


def compute_metrics(prob, labels):
    """prob: P(standard plane) per frame; labels: 0/1."""
    prob, labels = np.asarray(prob), np.asarray(labels)
    pred = (prob >= 0.5).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return {
        'acc': (tp + tn) / max(len(labels), 1),
        'balanced_acc': (recall + specificity) / 2.0,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': 2 * precision * recall / max(precision + recall, 1e-12),
        'auroc': auroc(prob, labels),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'num_frames': int(len(labels)),
        'majority_class_acc': float(max(labels.mean(), 1 - labels.mean())),
    }


# --------------------------------------------------------------------------- #
#  Linear probe
# --------------------------------------------------------------------------- #

@torch.no_grad()
def predict(head, x):
    return head(x).softmax(dim=-1)[:, 1].cpu().numpy()


def train_probe(train, val, args, device):
    """Train one linear classifier per weight decay; keep the (weight decay, epoch) best on val."""
    mean, std = train['x'].mean(dim=0, keepdim=True), train['x'].std(dim=0, keepdim=True) + 1e-6
    xtr, ytr = ((train['x'] - mean) / std).to(device), train['y'].to(device)
    xva, yva = ((val['x'] - mean) / std).to(device), val['y'].numpy()

    # -- class-balanced loss: train has more SP than NSP videos
    counts = torch.bincount(ytr, minlength=2).float()
    class_weight = (counts.sum() / (2.0 * counts)).to(device)

    best = {'score': -1.0}
    for wd in args.probe_wds:
        torch.manual_seed(args.seed)
        head = torch.nn.Linear(xtr.size(1), 2).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=args.probe_lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.probe_epochs * ((len(xtr) - 1) // args.probe_batch_size + 1))
        for epoch in range(1, args.probe_epochs + 1):
            perm = torch.randperm(len(xtr), device=device)
            for b in range(0, len(xtr), args.probe_batch_size):
                sel = perm[b:b + args.probe_batch_size]
                loss = F.cross_entropy(head(xtr[sel]), ytr[sel], weight=class_weight)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
            score = compute_metrics(predict(head, xva), yva)[args.select_metric]
            if score > best['score']:
                best = {'score': score, 'wd': wd, 'epoch': epoch,
                        'state': {k: v.clone() for k, v in head.state_dict().items()}}
        print(f'  weight decay {wd:g}: best val {args.select_metric} so far {best["score"]:.4f} '
              f'(wd={best["wd"]:g}, epoch={best["epoch"]})', flush=True)

    head = torch.nn.Linear(xtr.size(1), 2).to(device)
    head.load_state_dict(best['state'])
    return head, (mean, std), {k: best[k] for k in ('wd', 'epoch', 'score')}


def main():
    args = parse_args()
    assert args.random_init or args.checkpoint, 'pass --checkpoint or --random_init'
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- 1. features
    cache = os.path.join(args.out_dir, 'features.pt')
    if os.path.exists(cache) and not args.recompute_features:
        print(f'Loading cached features from {cache}')
        data = torch.load(cache)
    else:
        encoder, epoch = load_encoder(args, device)
        print('Extracting features...')
        data = {
            'epoch': epoch,
            'train': extract_train_features(encoder, read_train_list(args.train_csv), args, device),
            'val': extract_frame_features(encoder, read_frame_labels(args.val_dir), args, device, 'val'),
            'test': extract_frame_features(encoder, read_frame_labels(args.test_dir), args, device, 'test'),
        }
        torch.save(data, cache)
        del encoder
    for split in ('train', 'val', 'test'):
        y = data[split]['y']
        print(f'{split}: {len(y)} samples, {int(y.sum())} SP / {int((1 - y).sum())} NSP')

    # -- 2-3. probe, selected on validation
    print(f'Training linear probe (selecting on val {args.select_metric})...')
    head, (mean, std), selected = train_probe(data['train'], data['val'], args, device)

    # -- 4. report
    results = {
        'checkpoint': 'random_init' if args.random_init else args.checkpoint,
        'checkpoint_key': args.checkpoint_key,
        'pretrain_epoch': data.get('epoch'),
        'selected': selected,
        'select_metric': args.select_metric,
    }
    for split in ('val', 'test'):
        x = ((data[split]['x'] - mean) / std).to(device)
        prob = predict(head, x)
        results[split] = compute_metrics(prob, data[split]['y'].numpy())
        with open(os.path.join(args.out_dir, f'predictions_{split}.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['video', 'frame', 'label', 'prob_standard_plane'])
            for (video, frame), label, p in zip(data[split]['meta'], data[split]['y'].tolist(), prob):
                writer.writerow([video, frame, label, f'{p:.6f}'])

    with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nResults ({results["checkpoint"]}, pretrain epoch {results["pretrain_epoch"]}; '
          f'probe wd={selected["wd"]:g}, epoch={selected["epoch"]})')
    keys = ['acc', 'balanced_acc', 'f1', 'precision', 'recall', 'specificity', 'auroc', 'majority_class_acc']
    print('split  ' + '  '.join(f'{k:>12}' for k in keys))
    for split in ('val', 'test'):
        print(f'{split:<5}  ' + '  '.join(f'{results[split][k]:>12.4f}' for k in keys))
    print(f'Saved to {args.out_dir}/results.json')


if __name__ == '__main__':
    main()
