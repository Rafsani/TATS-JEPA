import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.utils.modules import Block
from src.models.utils.patch_embed import PatchEmbed3D
from src.models.utils.pos_embs import get_3d_sincos_pos_embed
from src.utils.tensors import trunc_normal_


class TrajectoryMaskPolicy(nn.Module):
    """
    Trajectory-aware masking policy.

    Scores every spatio-temporal patch of a clip with a learned "where is it
    worth masking" logit that combines (a) a lightweight transformer over the
    clip's own patch embeddings and (b) a per-patch motion-energy signal
    derived from consecutive-frame pixel differences within each tubelet
    (a proxy for the trajectories that pass through that patch, in the
    spirit of TATS/rl_videomae's trajectory-attention masking, without
    requiring an optical-flow dependency). The combined logits define a
    categorical distribution over patch locations that is used to *place*
    the prediction blocks (see `sample_trajectory_masks`), and is trained
    with REINFORCE using the V-JEPA prediction loss as (negative) reward.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        in_chans=3,
        embed_dim=256,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        uniform_power=False,
    ):
        super().__init__()
        self.input_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.grid_size = self.input_size // self.patch_size
        self.grid_depth = self.num_frames // self.tubelet_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.grid_size * self.grid_size * self.grid_depth, embed_dim),
            requires_grad=False,
        )
        self._init_pos_embed(self.pos_embed.data, uniform_power=uniform_power)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                grid_size=self.grid_size,
                grid_depth=self.grid_depth,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        # -- one scalar "worth masking" logit per patch (not a pooled
        # per-clip choice): this is what makes placement trajectory- and
        # sample-specific instead of a single decision shared by the batch.
        self.head = nn.Linear(embed_dim, 1)

        # -- learned weight on the motion-energy term; initialized > 0 so
        # the trajectory signal actually influences placement from the
        # start of training instead of having to be discovered from scratch.
        self.motion_scale = nn.Parameter(torch.tensor(1.0))

        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed, uniform_power=False):
        sincos = get_3d_sincos_pos_embed(
            pos_embed.size(-1),
            self.grid_size,
            self.grid_depth,
            cls_token=False,
            uniform_power=uniform_power,
        )
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        for layer_id, layer in enumerate(self.blocks):
            scale = math.sqrt(2.0 * (layer_id + 1))
            layer.attn.proj.weight.data.div_(scale)
            layer.mlp.fc2.weight.data.div_(scale)

    def _appearance_logits(self, x):
        """Per-patch appearance logits. Returns [B, N] (N = grid_depth*grid_size**2)."""
        x = self.patch_embed(x)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.head(x).squeeze(-1)

    def forward(self, clips):
        """
        Combine learned appearance logits with a motion-energy trajectory
        signal to produce the per-patch placement logits.

        NOTE: everything the policy learns (including `motion_scale`) is
        computed inside this single `forward` call, and nowhere else, so
        that a plain `mask_policy(clips)` call is sufficient to correctly
        engage DistributedDataParallel's gradient sync -- calling any other
        method on a DDP-wrapped module bypasses DDP's backward hooks.

        Returns:
            logits: [B, N] combined placement logits
            motion: [B, N] normalized motion-energy signal (for logging/debug)
        """
        appearance_logits = self._appearance_logits(clips)
        motion = compute_motion_energy(
            clips, grid_depth=self.grid_depth, grid_size=self.grid_size,
            patch_size=self.patch_size,
        )
        motion = motion.flatten(1)  # [B, N], same (d, h, w) flatten order as appearance_logits
        motion = _standardize(motion)
        logits = appearance_logits + self.motion_scale * motion
        return logits, motion


def compute_motion_energy(clips, grid_depth, grid_size, patch_size):
    """
    Cheap trajectory-saliency proxy: how much each spatio-temporal patch's
    content changes across the clip, pooled to the patch grid.

    clips: [B, C, T, H, W]
    Returns: [B, grid_depth, grid_size, grid_size]
    """
    B, C, T, H, W = clips.shape

    if T > 1:
        # -- temporal motion energy: mean abs frame-to-frame difference
        frame_diff = (clips[:, :, 1:] - clips[:, :, :-1]).abs().mean(dim=1)  # [B, T-1, H, W]
    else:
        # -- degenerate single-frame clip: fall back to spatial-gradient
        # magnitude as a static-saliency proxy so the signal is never empty.
        gx = F.pad((clips[:, :, :, :, 1:] - clips[:, :, :, :, :-1]).abs(), (0, 1))
        gy = F.pad((clips[:, :, :, 1:, :] - clips[:, :, :, :-1, :]).abs(), (0, 0, 0, 1))
        frame_diff = (gx + gy).mean(dim=1)  # [B, T, H, W]

    Bp, Tp = frame_diff.shape[0], frame_diff.shape[1]
    pooled = F.avg_pool2d(
        frame_diff.reshape(Bp * Tp, 1, H, W),
        kernel_size=patch_size, stride=patch_size,
    ).reshape(Bp, Tp, grid_size, grid_size)  # [B, Tp, gh, gw]

    if Tp == grid_depth:
        motion = pooled
    else:
        # -- pool/interpolate whatever number of frame-diffs we have onto
        # the grid_depth tubelet buckets.
        motion = F.adaptive_avg_pool1d(
            pooled.permute(0, 2, 3, 1).reshape(Bp * grid_size * grid_size, 1, Tp),
            grid_depth,
        ).reshape(Bp, grid_size, grid_size, grid_depth).permute(0, 3, 1, 2)

    return motion  # [B, grid_depth, grid_size, grid_size]


def _standardize(x, eps=1e-5):
    """Per-sample zero-mean/unit-std normalization over the last dim."""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + eps)


def sample_block_size(seed, temporal_scale, spatial_scale, aspect_ratio, duration, height, width):
    """
    Sample a (t, h, w) block shape for a mask spec, deterministically given
    `seed` (mirrors `multiblock3d.MaskGenerator._sample_block_size`).
    """
    g = torch.Generator()
    g.manual_seed(seed)

    _rand = torch.rand(1, generator=g).item()
    min_t, max_t = temporal_scale
    temporal_mask_scale = min_t + _rand * (max_t - min_t)
    t = max(1, int(duration * temporal_mask_scale))

    _rand = torch.rand(1, generator=g).item()
    min_s, max_s = spatial_scale
    spatial_mask_scale = min_s + _rand * (max_s - min_s)
    spatial_num_keep = int(height * width * spatial_mask_scale)

    _rand = torch.rand(1, generator=g).item()
    min_ar, max_ar = aspect_ratio
    ar = min_ar + _rand * (max_ar - min_ar)

    h = int(round(math.sqrt(spatial_num_keep * ar)))
    w = int(round(math.sqrt(spatial_num_keep / ar)))
    h = min(max(h, 1), height)
    w = min(max(w, 1), width)

    return t, h, w


def sample_trajectory_masks(policy_logits, grid_shape, block_specs):
    """
    Sample encoder/predictor masks whose block *placement* is driven by
    `policy_logits`, instead of the uniformly-random placement used by the
    stock multiblock3d masking.

    Args:
        policy_logits: [B, N] trajectory-aware per-patch logits (N = D*H*W,
            flattened in (depth, height, width) row-major order).
        grid_shape: (duration, height, width) of the patch grid.
        block_specs: list of dicts, one per mask config, each with keys
            't', 'h', 'w' (block shape), 'npred' (number of blocks),
            'max_context_duration', 'max_keep' (optional).

    Returns:
        masks_enc: list[Tensor[B, min_keep_enc]] (one per config)
        masks_pred: list[Tensor[B, min_keep_pred]] (one per config)
        log_prob: Tensor[B] summed log-prob of every block placement sampled
            this step (for REINFORCE).
        entropy: Tensor[B] entropy of the per-sample placement distribution.
    """
    D, H, W = grid_shape
    B, N = policy_logits.shape
    assert N == D * H * W, f'grid_shape {grid_shape} does not match policy_logits width {N}'
    device = policy_logits.device

    dist = torch.distributions.Categorical(logits=policy_logits)
    entropy = dist.entropy()  # [B]
    log_prob = torch.zeros(B, device=device)

    masks_enc, masks_pred = [], []
    for spec in block_specs:
        t, h, w = spec['t'], spec['h'], spec['w']
        npred = max(1, int(spec.get('npred', 1)))
        max_context_duration = spec.get('max_context_duration', D)
        max_keep = spec.get('max_keep', None)

        mask_e = torch.ones(B, D, H, W, dtype=torch.int32, device=device)
        for _ in range(npred):
            anchor = dist.sample()  # [B]
            log_prob = log_prob + dist.log_prob(anchor)

            d0 = torch.div(anchor, H * W, rounding_mode='floor')
            rem = anchor % (H * W)
            h0 = torch.div(rem, W, rounding_mode='floor')
            w0 = rem % W

            start = (d0 - t // 2).clamp(0, max(D - t, 0))
            top = (h0 - h // 2).clamp(0, max(H - h, 0))
            left = (w0 - w // 2).clamp(0, max(W - w, 0))

            for b in range(B):
                s, tp, lf = int(start[b]), int(top[b]), int(left[b])
                mask_e[b, s:s + t, tp:tp + h, lf:lf + w] = 0

        if max_context_duration < D:
            mask_e[:, max_context_duration:, :, :] = 0

        mask_e_flat = mask_e.reshape(B, -1)

        enc_rows, pred_rows = [], []
        min_keep_enc = min_keep_pred = D * H * W
        for b in range(B):
            row = mask_e_flat[b]
            enc_idx = torch.nonzero(row, as_tuple=False).squeeze(-1)
            if enc_idx.numel() == 0:
                # -- degenerate case (block(s) covered the whole clip):
                # force-keep the single highest-logit patch as context so
                # training never sees an empty context.
                row = row.clone()
                row[policy_logits[b].argmax()] = 1
                enc_idx = torch.nonzero(row, as_tuple=False).squeeze(-1)
            pred_idx = torch.nonzero(row == 0, as_tuple=False).squeeze(-1)

            enc_rows.append(enc_idx)
            pred_rows.append(pred_idx)
            min_keep_enc = min(min_keep_enc, enc_idx.numel())
            min_keep_pred = min(min_keep_pred, pred_idx.numel())

        if max_keep is not None:
            min_keep_enc = min(min_keep_enc, max_keep)
        min_keep_pred = max(min_keep_pred, 1)
        min_keep_enc = max(min_keep_enc, 1)

        masks_enc.append(torch.stack([e[:min_keep_enc] for e in enc_rows]))
        masks_pred.append(torch.stack([p[:min_keep_pred] for p in pred_rows]))

    return masks_enc, masks_pred, log_prob, entropy


class NoOpMaskCollator:
    """
    Collate function used when `rl_mask.enabled: true`: masks are generated
    on-GPU from the trajectory policy once the training loop can see the
    actual clip content, so the dataloader only needs to collate clips.
    A plain (picklable, module-level) class rather than a closure, since
    DataLoader workers pickle the collate_fn when num_workers > 0.
    """

    def __call__(self, batch):
        return torch.utils.data.default_collate(batch), None, None


def unwrap_policy(module):
    if hasattr(module, 'module'):
        module = module.module
    return module
