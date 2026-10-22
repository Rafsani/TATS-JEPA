"""
GRPO (Group Relative Policy Optimization) masking for V-JEPA.

Same actor as the TATS/PPO version (src/masks/tats_ppo.py: trajectory
attention over the encoder's patch embeddings, one logit per token), but
trained without a critic:

  * for every clip, sample a GROUP of G masks from the policy
    (each mask = `num_visible` context tokens drawn without replacement)
  * V-JEPA trains on all G masks; each mask's reward is its mean prediction
    loss (harder masks score higher, as in TATS)
  * advantage of a mask = (reward - group mean) / group std, so masks are
    only compared with other masks of the *same clip*
  * the policy update is the clipped PPO surrogate, applied to every pick of
    the mask (the mask is a sequence of K picks, like the tokens of an LLM
    completion in the original GRPO)

Log-probabilities are exact for sampling without replacement
(Plackett-Luce): pick k of a mask has probability
    p(i_k) / (1 - sum_{j<k} p(i_j)),
which requires the picks in the order torch.multinomial drew them.
"""

import torch

from src.masks.tats_ppo import policy_entropy, policy_log_probs, split_masks


@torch.no_grad()
def sample_group_masks(logits, num_visible, group_size, explore_eps=0.0):
    """
    logits: [B, N] policy logits for B clips.
    Returns:
        vis_idx: [G, B, K] context tokens per mask, in draw order
        masks_enc: list of G tensors [B, K] (sorted context indices)
        masks_pred: list of G tensors [B, N - K] (sorted target indices)
    """
    B, N = logits.shape
    probs = policy_log_probs(logits, explore_eps).exp()
    vis_idx = torch.multinomial(probs.repeat(group_size, 1), num_visible, replacement=False)
    vis_idx = vis_idx.view(group_size, B, num_visible)
    masks_enc, masks_pred = zip(*[split_masks(vis_idx[g], N) for g in range(group_size)])
    return vis_idx, list(masks_enc), list(masks_pred)


@torch.no_grad()
def sample_random_group_masks(batch_size, num_patches, num_visible, group_size, device):
    """Uniformly random group of masks (used during the policy warmup)."""
    vis_idx = torch.rand(group_size * batch_size, num_patches, device=device).argsort(dim=-1)[:, :num_visible]
    vis_idx = vis_idx.view(group_size, batch_size, num_visible)
    masks_enc, masks_pred = zip(*[split_masks(vis_idx[g], num_patches) for g in range(group_size)])
    return list(masks_enc), list(masks_pred)


def pick_log_probs(logits, vis_idx, explore_eps=0.0):
    """
    Exact log-probability of every pick of every mask (Plackett-Luce).

    logits: [B, N]; vis_idx: [G, B, K] in draw order -> [G, B, K]
    """
    G = vis_idx.size(0)
    log_p = policy_log_probs(logits, explore_eps)  # [B, N]
    log_p_sel = log_p.unsqueeze(0).expand(G, -1, -1).gather(2, vis_idx)  # [G, B, K]
    p_sel = log_p_sel.exp()
    taken_before = torch.cumsum(p_sel, dim=-1) - p_sel  # probability mass already drawn
    return log_p_sel - torch.log1p(-taken_before.clamp(max=1.0 - 1e-6))


def group_advantages(rewards, scale_by_std=True, eps=1e-6):
    """rewards: [G, B] -> advantages [G, B], normalised within each clip's group."""
    adv = rewards - rewards.mean(dim=0, keepdim=True)
    if scale_by_std:
        adv = adv / (rewards.std(dim=0, keepdim=True) + eps)
    return adv


def grpo_update(
    policy,
    optimizer,
    state,
    vis_idx,
    old_log_probs,
    advantages,
    num_iterations=1,
    clip_eps=0.2,
    entropy_coef=0.0,
    kl_coef=0.0,
    explore_eps=0.0,
    clip_grad=None,
    amp_dtype=None,
):
    """
    Clipped GRPO update on one group rollout.

    state: [B, N, D] policy input; vis_idx / old_log_probs: [G, B, K];
    advantages: [G, B]. `policy` may be DDP-wrapped: the skip decision below
    uses the all-reduced gradient norm, so it is identical on every rank.
    """
    totals = {'policy_loss': 0., 'entropy': 0., 'clip_frac': 0., 'approx_kl': 0.}
    num_updates, num_skipped = 0, 0
    adv = advantages.unsqueeze(-1)  # [G, B, 1]: every pick of a mask shares its advantage

    for _ in range(num_iterations):
        with torch.autocast('cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
            logits, _ = policy(state)
        new_log_probs = pick_log_probs(logits, vis_idx, explore_eps)

        log_ratio = (new_log_probs - old_log_probs).clamp(-20.0, 20.0)
        ratio = log_ratio.exp()
        surr = torch.min(ratio * adv, ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv)
        policy_loss = -surr.mean()
        entropy = policy_entropy(logits, explore_eps).mean()
        # -- k3 estimator of KL(old || new); 0 when kl_coef = 0
        kl = (torch.exp(-log_ratio) + log_ratio - 1.0).mean()
        loss = policy_loss - entropy_coef * entropy + kl_coef * kl

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), clip_grad if clip_grad is not None else float('inf'))
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad()
            num_skipped += 1
            continue
        optimizer.step()

        with torch.no_grad():
            totals['policy_loss'] += float(policy_loss)
            totals['entropy'] += float(entropy)
            totals['clip_frac'] += float(((ratio - 1.0).abs() > clip_eps).float().mean())
            totals['approx_kl'] += float(kl)
        num_updates += 1

    out = {k: v / max(num_updates, 1) for k, v in totals.items()}
    out['skipped'] = num_skipped
    return out
