"""
TATS (Trajectory-Aware Adaptive Token Sampler) + PPO masking for V-JEPA.

Port of the masking policy from "Reinforcement Learning meets Masked Video
Modeling: Trajectory-Guided Adaptive Token Selection" (Rai et al., ICCVW 2025,
https://github.com/rayush7/rl_videomae), adapted from VideoMAE to V-JEPA:

  * state   = the context encoder's (detached) patch embeddings of the clip
  * policy  = trajectory-attention block(s) (Motionformer) -> 1 logit / token
  * action  = sample `num_visible` context tokens without replacement from
              softmax(logits); every other token is a prediction target
  * reward  = per-target-token V-JEPA prediction loss (high loss = high reward)
  * update  = PPO: clipped surrogate on the per-token log-probs of the
              target tokens, MSE value loss, entropy bonus

Deviations from the reference code (all deliberate, see comments inline):
  * PPO re-evaluates log-probs at the *stored* actions; the reference
    resamples a fresh mask inside the PPO update, so its ratio compared
    log-probs of different tokens.
  * The policy runs on its own optimizer instead of sharing the MAE one.
  * Optional input projection (encoder dim -> policy dim) to keep the
    trajectory attention affordable next to a ViT-L encoder.
  * The learnable policy position embedding is initialised from 3D sin-cos
    instead of zeros.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.utils.modules import MLP, Block
from src.models.utils.pos_embs import get_3d_sincos_pos_embed


# --------------------------------------------------------------------------- #
#  Trajectory attention (Motionformer, Patrick et al. 2021)
# --------------------------------------------------------------------------- #

def orthogonal_landmarks(q, num_landmarks):
    """
    Greedily pick `num_landmarks` queries that are maximally orthogonal to the
    ones already picked (Orthoformer landmark selection).

    q: [B, N, D] -> landmarks [B, L, D]
    """
    B, N, _ = q.shape
    num_landmarks = min(num_landmarks, N)
    qn = F.normalize(q.float(), p=2, dim=-1)
    batch = torch.arange(B, device=q.device)

    selected = torch.zeros(B, N, dtype=torch.bool, device=q.device)
    # -- deterministic start (largest-norm query) instead of Orthoformer's
    # random one: PPO needs the rollout and the update to see the same logits
    idx = q.float().norm(dim=-1).argmax(dim=-1)
    selected[batch, idx] = True
    picked = [idx]
    # -- running max |cos| between every query and the selected set
    max_cos = torch.zeros(B, N, device=q.device)
    for _ in range(1, num_landmarks):
        cos = torch.einsum('bnd,bd->bn', qn, qn[batch, idx]).abs()
        max_cos = torch.maximum(max_cos, cos)
        idx = max_cos.masked_fill(selected, float('inf')).argmin(dim=-1)
        selected[batch, idx] = True
        picked.append(idx)

    picked = torch.stack(picked, dim=1)  # [B, L]
    return q[batch.unsqueeze(1), picked]  # [B, L, D]


def orthoformer_attention(q, k, v, num_frames, num_landmarks=128):
    """
    Landmark approximation of the per-frame spatial attention used by
    trajectory attention.

    q, k, v: [B', N, d] with tokens ordered (frame, patch)
    Returns: [B', N, F, d] -- for every query, its attention-weighted
        aggregate of each frame's values (the "trajectory" through time).
    """
    BH, N, d = q.shape
    Fr = num_frames
    P = N // Fr

    scale = d ** -0.25
    q = q * scale
    k = k * scale

    with torch.no_grad():
        landmarks = orthogonal_landmarks(q, num_landmarks)
    L = landmarks.size(1)

    kernel_1 = F.softmax(q @ landmarks.transpose(-1, -2), dim=-1)  # [B', N, L]
    kernel_2 = F.softmax(
        (landmarks @ k.transpose(-1, -2)).view(BH, L, Fr, P), dim=-1)  # [B', L, F, P]
    x = torch.einsum('blfp,bfpd->blfd', kernel_2, v.view(BH, Fr, P, d))
    x = torch.einsum('bnl,blfd->bnfd', kernel_1, x)
    return x


def full_trajectory_space_attention(q, k, v, num_frames, scale):
    """Exact (quadratic) version of `orthoformer_attention`."""
    BH, N, d = q.shape
    Fr = num_frames
    P = N // Fr
    attn = ((q @ k.transpose(-2, -1)) * scale).view(BH, N, Fr, P).softmax(dim=-1)
    return torch.einsum('bqfp,bfpd->bqfd', attn, v.view(BH, Fr, P, d))


class TrajectoryAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        proj_drop=0.,
        approx='orthoformer',
        num_landmarks=128,
    ):
        super().__init__()
        assert approx in ('orthoformer', 'none'), f'unknown trajectory attention approx {approx}'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.approx = approx
        self.num_landmarks = num_landmarks

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, num_frames):
        B, N, C = x.shape
        h, d, Fr = self.num_heads, self.head_dim, num_frames
        P = N // Fr
        assert Fr * P == N, f'{N} tokens cannot be split into {Fr} frames'

        # -- step 1: per-frame spatial attention -> one aggregate per (query, frame)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(B, N, h, d).transpose(1, 2).reshape(B * h, N, d) for t in (q, k, v)]
        if self.approx == 'orthoformer':
            x = orthoformer_attention(q, k, v, Fr, self.num_landmarks)
        else:
            x = full_trajectory_space_attention(q, k, v, Fr, self.scale)
        x = x.reshape(B, h, N, Fr, d).permute(0, 2, 3, 1, 4).reshape(B, N, Fr, C)  # [B, N, F, C]

        # -- step 2: temporal attention along each token's trajectory; the
        # query is the token's own-frame aggregate
        f_idx = torch.arange(Fr, device=x.device)
        x_diag = x.view(B, Fr, P, Fr, C)[:, f_idx, :, f_idx]  # [F, B, P, C]
        x_diag = x_diag.permute(1, 0, 2, 3).reshape(B, N, C)
        q2 = self.proj_q(x_diag).reshape(B, N, h, d).transpose(1, 2) * self.scale  # [B, h, N, d]
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1)
        k2, v2 = [t.reshape(B, N, Fr, h, d).permute(0, 3, 1, 2, 4) for t in (k2, v2)]  # [B, h, N, F, d]
        attn = torch.einsum('bhsd,bhsfd->bhsf', q2, k2).softmax(dim=-1)
        x = torch.einsum('bhsf,bhsfd->bhsd', attn, v2)
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TrajBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        drop=0.,
        approx='orthoformer',
        num_landmarks=128,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TrajectoryAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_drop=drop,
            approx=approx,
            num_landmarks=num_landmarks)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop)

    def forward(self, x, num_frames):
        x = x + self.attn(self.norm1(x), num_frames)
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------------------------------- #
#  Policy / value network
# --------------------------------------------------------------------------- #

class TATSPolicy(nn.Module):
    """
    Trajectory-aware adaptive token sampler with a value head.

    forward(state) -> (logits [B, N], value [B])
      state: [B, N, in_dim] patch embeddings, tokens ordered (t, h, w)
    """

    def __init__(
        self,
        in_dim,
        grid_depth,
        grid_size,
        embed_dim=384,
        traj_attn=True,
        num_blocks=1,
        num_heads=8,
        mlp_ratio=4.,
        qkv_bias=False,
        drop=0.,
        approx='orthoformer',
        num_landmarks=128,
        softmax_temp=1.0,
        uniform_power=True,
    ):
        super().__init__()
        self.grid_depth = grid_depth
        self.grid_size = grid_size
        self.num_patches = grid_depth * grid_size * grid_size
        self.traj_attn = traj_attn
        self.softmax_temp = softmax_temp

        # -- normalise the encoder's patch embeddings, whose scale grows during
        # training; without it the untrained policy starts out near-deterministic
        self.input_norm = nn.LayerNorm(in_dim)
        self.input_proj = nn.Linear(in_dim, embed_dim) if in_dim != embed_dim else nn.Identity()
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        if traj_attn:
            self.blocks = nn.ModuleList([
                TrajBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    approx=approx,
                    num_landmarks=num_landmarks)
                for _ in range(num_blocks)])
        else:
            # -- plain joint space-time self-attention (TATS' non-trajectory ablation)
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop)
                for _ in range(num_blocks)])
        self.head = nn.Linear(embed_dim, 1)

        # -- critic: MLP over the per-token channel-mean of the state (as in TATS)
        self.value_net = nn.Sequential(
            nn.Linear(self.num_patches, self.num_patches // 2),
            nn.ReLU(),
            nn.Linear(self.num_patches // 2, 1))

        self.apply(self._init_weights)
        sincos = get_3d_sincos_pos_embed(
            embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=uniform_power)
        self.pos_embed.data.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, state):
        x = self.input_proj(self.input_norm(state)) + self.pos_embed
        value = self.value_net(x.mean(dim=-1)).squeeze(-1)

        for blk in self.blocks:
            x = blk(x, self.grid_depth) if self.traj_attn else blk(x)
        logits = self.head(x).squeeze(-1) / self.softmax_temp

        return logits.float(), value.float()


# --------------------------------------------------------------------------- #
#  Masks, log-probs, PPO loss
# --------------------------------------------------------------------------- #

def split_masks(vis_idx, num_patches):
    """
    vis_idx: [B, K] context-token indices
    Returns (masks_enc [B, K], masks_pred [B, N-K]), each sorted ascending.
    """
    B = vis_idx.size(0)
    keep = torch.zeros(B, num_patches, dtype=torch.bool, device=vis_idx.device)
    keep.scatter_(1, vis_idx, True)
    ids = torch.arange(num_patches, device=vis_idx.device).expand(B, num_patches)
    return ids[keep].view(B, -1), ids[~keep].view(B, -1)


def policy_log_probs(logits, explore_eps=0.0):
    """
    log of the sampling distribution (1 - eps) * softmax(logits) + eps / N.

    The uniform floor bounds every token's log-prob below by log(eps / N), so
    a sharpening policy cannot drive the PPO ratio exp(new - old) to overflow.
    """
    log_p = F.log_softmax(logits.float(), dim=-1)
    if explore_eps <= 0:
        return log_p
    N = logits.size(-1)
    return torch.logaddexp(
        log_p + math.log(1.0 - explore_eps),
        torch.full_like(log_p, math.log(explore_eps / N)))


@torch.no_grad()
def sample_tats_masks(logits, num_visible, explore_eps=0.0):
    """Sample context tokens from the policy distribution without replacement."""
    probs = policy_log_probs(logits, explore_eps).exp()
    vis_idx = torch.multinomial(probs, num_visible, replacement=False)
    return split_masks(vis_idx, logits.size(-1))


@torch.no_grad()
def sample_random_masks(batch_size, num_patches, num_visible, device):
    """Uniformly random context tokens (used during the policy warmup)."""
    vis_idx = torch.rand(batch_size, num_patches, device=device).argsort(dim=-1)[:, :num_visible]
    return split_masks(vis_idx, num_patches)


def token_log_probs(logits, idx, explore_eps=0.0):
    """log pi(token) of each token index in `idx` ([B, K]) under the policy distribution."""
    return policy_log_probs(logits, explore_eps).gather(1, idx)


def policy_entropy(logits, explore_eps=0.0):
    log_p = policy_log_probs(logits, explore_eps)
    return -(log_p.exp() * log_p).sum(dim=-1)


def ppo_loss(
    new_log_probs,
    new_values,
    old_log_probs,
    old_values,
    rewards,
    clip_eps=0.2,
    dual_clip=3.0,
    normalize_advantage=False,
):
    """
    Per-token clipped PPO objective (TATS formulation), with dual-clip.

    new_log_probs, old_log_probs, rewards: [B, N_pred]
    new_values, old_values: [B]
    """
    advantages = rewards - old_values.unsqueeze(-1)
    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # -- clamp the log-ratio so a single token cannot overflow exp()
    ratio = torch.exp((new_log_probs - old_log_probs).clamp(-20.0, 20.0))
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    surr = torch.min(surr1, surr2)
    if dual_clip is not None:
        # -- dual-clip PPO (Ye et al. 2020): for negative advantages the plain
        # clipped objective is unbounded in the ratio; cap it at dual_clip * A
        surr = torch.where(advantages < 0, torch.max(surr, dual_clip * advantages), surr)
    policy_loss = -surr.mean()

    # -- critic regresses the per-token rewards of its clip (TATS broadcasts [B, 1] -> [B, N_pred])
    value_loss = F.mse_loss(new_values.unsqueeze(-1).expand_as(rewards), rewards)

    with torch.no_grad():
        stats = {
            'clip_frac': float(((ratio - 1.0).abs() > clip_eps).float().mean()),
            'approx_kl': float((old_log_probs - new_log_probs).mean()),
        }
    return policy_loss, value_loss, stats


class PPORolloutBuffer:
    """Stores (state, action, old log-prob, old value, reward) per training step."""

    def __init__(self, store_on_cpu=False):
        self.store_on_cpu = store_on_cpu
        self.items = []

    def add(self, **kwargs):
        item = {}
        for k, v in kwargs.items():
            v = v.detach()
            item[k] = v.cpu() if self.store_on_cpu else v
        self.items.append(item)

    def clear(self):
        self.items = []

    def __len__(self):
        return len(self.items)


def _mean_across_ranks(x):
    x = torch.tensor(float(x), device='cuda' if torch.cuda.is_available() else 'cpu')
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(x)
        x /= torch.distributed.get_world_size()
    return float(x)


def ppo_update(
    policy,
    optimizer,
    buffer,
    device,
    ppo_epochs=1,
    minibatch_size=None,
    clip_eps=0.2,
    dual_clip=3.0,
    policy_coef=1.0,
    value_coef=1.0,
    entropy_coef=0.0,
    normalize_advantage=False,
    explore_eps=0.0,
    clip_grad=None,
    target_kl=None,
    amp_dtype=None,
):
    """
    Run `ppo_epochs` passes over every rollout in `buffer`.

    `policy` may be DDP-wrapped: every rank must hold the same number of
    rollouts with the same batch size so the number of backward passes (and
    therefore gradient all-reduces) matches across ranks. The skip / early-stop
    decisions below are therefore taken on rank-synchronised quantities.
    """
    totals = {'policy_loss': 0., 'value_loss': 0., 'entropy': 0., 'clip_frac': 0., 'approx_kl': 0.}
    num_updates = 0
    num_skipped = 0
    early_stopped = False

    for _ in range(ppo_epochs):
        for i in torch.randperm(len(buffer)).tolist():
            item = buffer.items[i]
            B = item['state'].size(0)
            mb = minibatch_size or B
            perm = torch.randperm(B)
            for s in range(0, B, mb):
                sel = perm[s:s + mb]
                state = item['state'][sel].to(device, non_blocking=True)
                actions = item['actions'][sel].to(device, non_blocking=True)
                old_log_probs = item['old_log_probs'][sel].to(device, non_blocking=True)
                old_values = item['old_values'][sel].to(device, non_blocking=True)
                rewards = item['rewards'][sel].to(device, non_blocking=True)

                with torch.autocast('cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
                    logits, values = policy(state)
                new_log_probs = token_log_probs(logits, actions, explore_eps)
                entropy = policy_entropy(logits, explore_eps).mean()
                p_loss, v_loss, stats = ppo_loss(
                    new_log_probs, values, old_log_probs, old_values, rewards,
                    clip_eps=clip_eps, dual_clip=dual_clip,
                    normalize_advantage=normalize_advantage)
                loss = policy_coef * p_loss + value_coef * v_loss - entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                # -- grads are all-reduced by now, so every rank sees the same norm
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), clip_grad if clip_grad is not None else float('inf'))
                if not torch.isfinite(grad_norm):
                    optimizer.zero_grad()
                    num_skipped += 1
                    continue
                optimizer.step()

                totals['policy_loss'] += float(p_loss)
                totals['value_loss'] += float(v_loss)
                totals['entropy'] += float(entropy)
                totals['clip_frac'] += stats['clip_frac']
                totals['approx_kl'] += stats['approx_kl']
                num_updates += 1

                if target_kl is not None and _mean_across_ranks(stats['approx_kl']) > 1.5 * target_kl:
                    early_stopped = True
                    break
            if early_stopped:
                break
        if early_stopped:
            break

    out = {k: v / max(num_updates, 1) for k, v in totals.items()}
    out['skipped'] = num_skipped
    out['early_stop'] = early_stopped
    return out
