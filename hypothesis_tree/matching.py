"""Cluster matching: the forward pass's competition primitive.

A cluster is a box in feature space: an anchor row, an active-feature mask,
and an asymmetric per-feature tolerance (how far below / above the anchor a
value may sit and still count as "inside"). A matcher scores a block of
samples against a shared set of candidate clusters and returns each sample's
winning cluster (-1 on a miss). Two strategies are built in (see
MATCH_STRATEGIES):

- "scored" (default, `batched_match_scored`): elastic-bound Gaussian — a
  sample inside the box scores 1.0; outside, the score decays as
  exp(-d / softness) where d is the L2 norm of the per-feature out-of-bounds
  distances. Effectively: the nearest box wins.
- "closer" (`batched_match_closer`): vote ratio — a candidate scores the
  fraction of its active features on which it is the closest eligible
  candidate to the sample. During descent, a child is eligible only if the
  sample lies within its tolerance on at least one active feature (the
  parent always bypasses this gate).

Both share the descent tie-break in `best_local_index`.
"""
import numpy as np


def best_local_index(scores, parent_pos):
    """Winning column per row of an (m, K) score matrix.

    parent_pos is the column holding the parent cluster during descent, or
    None for the root competition.

    Root: plain per-row argmax (ties: lowest column index wins).
    Descent: two-phase tie-break — the best child wins if it at least ties the
    parent (so a child can capture rows away from its parent), and among
    children the oldest (lowest index) wins ties. If no child ties or beats
    the parent, the parent keeps the row.
    """
    if parent_pos is None:
        return np.argmax(scores, axis=1)

    rows = np.arange(scores.shape[0])
    children_only = scores.copy()
    children_only[:, parent_pos] = -np.inf
    best_child = np.argmax(children_only, axis=1)
    child_score = children_only[rows, best_child]
    parent_score = scores[:, parent_pos]

    child_wins = (child_score >= parent_score) & (child_score >= 0)
    parent_wins = ~child_wins & (parent_score >= 0)
    fallback = np.argmax(scores, axis=1)  # everyone scored < 0: caller flags a miss
    return np.where(child_wins, best_child, np.where(parent_wins, parent_pos, fallback))


def batched_match_scored(samples, mem_X, mem_masks, mem_atol, cluster_indices,
                         parent_pos=None, softness=0.1):
    """Match M samples against K candidate clusters; return per-sample winners.

    samples: (M, D). cluster_indices: K global cluster indices, shared by all
    samples in the block. parent_pos: local position of the parent within
    cluster_indices during descent (None = root competition). Returns an (M,)
    int array of global winning cluster indices, -1 on a miss.
    """
    n_samples = samples.shape[0]
    if len(cluster_indices) == 0:
        return np.full(n_samples, -1, dtype=int)

    cand = np.asarray(cluster_indices)
    box_X = mem_X[cand]          # (K, D)
    box_mask = mem_masks[cand]   # (K, D)
    box_atol = mem_atol[cand]    # (K, 2, D)

    # Fold the active-feature mask into the bounds: inactive features get
    # (-inf, +inf) so their out-of-bounds distance is exactly 0.
    low = box_X - box_atol[:, 0, :]
    high = box_X + box_atol[:, 1, :]
    low[~box_mask] = -np.inf
    high[~box_mask] = np.inf
    eligible = np.any(box_mask, axis=1)  # (K,) needs >=1 active feature

    s = samples[:, None, :]                                   # (M, 1, D)
    oob = np.maximum(0.0, np.maximum(low[None] - s, s - high[None]))
    distance = np.sqrt(np.sum(oob * oob, axis=2))             # (M, K)
    scores = np.where(distance == 0, 1.0, np.exp(-distance / softness))
    scores = np.where(eligible[None, :], scores, -1.0)

    best = best_local_index(scores, parent_pos)
    best_score = scores[np.arange(n_samples), best]
    return np.where(best_score < 0, -1, cand[best])


def batched_match_closer(samples, mem_X, mem_masks, mem_atol, cluster_indices,
                         parent_pos=None, softness=None):
    """Vote-ratio matcher (the "closer" strategy). Same signature and return
    contract as `batched_match_scored`; `softness` is accepted for a uniform
    signature but unused.

    Per feature, the candidate whose anchor is closest to the sample (within
    1e-9) gets a vote; a candidate's score is its votes over its active
    feature count. During descent, candidates are gated by their tolerance
    (the sample must be inside on at least one active feature); the parent
    bypasses the gate. The root competition is ungated.
    """
    n_samples = samples.shape[0]
    if len(cluster_indices) == 0:
        return np.full(n_samples, -1, dtype=int)

    cand = np.asarray(cluster_indices)
    box_X = mem_X[cand]          # (K, D)
    box_mask = mem_masks[cand]   # (K, D)
    box_atol = mem_atol[cand]    # (K, 2, D)
    lower_tol = box_atol[:, 0, :]
    upper_tol = box_atol[:, 1, :]
    active = box_mask.sum(axis=1)  # (K,) vote-ratio denominator

    signed = samples[:, None, :] - box_X[None]   # (M, K, D)
    abs_diffs = np.abs(signed)

    if parent_pos is None:
        eligible = np.ones((n_samples, len(cand)), dtype=bool)
    else:
        within = (((signed < 0) & (abs_diffs <= lower_tol[None] + 1e-9))
                  | ((signed >= 0) & (abs_diffs <= upper_tol[None] + 1e-9)))
        eligible = np.any(within & box_mask[None], axis=2)  # (M, K)
        eligible[:, parent_pos] = True  # parent bypasses the gate

    masked = np.where(eligible[:, :, None], abs_diffs, np.inf)
    closest = masked.min(axis=1)                             # (M, D)
    in_bounds = (abs_diffs <= closest[:, None, :] + 1e-9) & eligible[:, :, None]
    votes = (in_bounds & box_mask[None]).sum(axis=2)         # (M, K)
    scores = votes / np.maximum(active[None], 1)
    scores = np.where(eligible, scores, -1.0)

    best = best_local_index(scores, parent_pos)
    best_score = scores[np.arange(n_samples), best]
    return np.where(best_score < 0, -1, cand[best])


MATCH_STRATEGIES = {
    "scored": batched_match_scored,
    "closer": batched_match_closer,
}


def get_match_fn(match_strategy):
    """Resolve a strategy name (or a callable with the same signature) into
    the matcher function."""
    if callable(match_strategy):
        return match_strategy
    try:
        return MATCH_STRATEGIES[match_strategy]
    except KeyError:
        raise ValueError(
            f"unknown match_strategy {match_strategy!r}; "
            f"expected one of {sorted(MATCH_STRATEGIES)} or a callable"
        ) from None
