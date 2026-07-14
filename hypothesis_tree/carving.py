"""Carving: turn a scored family of errors into a concrete child cluster.

Given the family box chosen by `scoring` (features + value ranges over the
misrouted samples' diffs), this module picks the cluster's anchor row, sizes
its per-feature tolerances, and applies the eat-parent gate — a routing
simulation that stops a child whose box would steal (almost) all of the
parent's correctly-routed samples.

`carve` is the entry point one carve iteration calls; `expand_atol_to_cover`
is the separate "grow a cluster's box to keep its goods inside" repair used
by the backward pass.
"""
import numpy as np

from .scoring import (
    EPS,
    build_family_intersection,
    family_quality,
    score_all_features,
)

# Minimum half-width of an active feature's tolerance (never exactly zero).
ATOL_FLOOR = 1e-7


def bound_scores(diffs, low_bound, high_bound, mask, softness):
    """Elastic-Gaussian match scores of diff rows against a [low, high] box on
    `mask` features — the same scoring rule the forward matcher uses, applied
    in diff space. Returns an (n,) score array."""
    oob = np.maximum(0.0, np.maximum(low_bound - diffs, diffs - high_bound)) * mask
    distance = np.sqrt(np.sum(oob ** 2, axis=1))
    return np.where(distance == 0.0, 1.0, np.exp(-distance / softness))


def count_parent_keeps(good_diffs, anchor_val, l_atol, h_atol, child_mask,
                       parent_atol, parent_mask, softness=0.1, gate_cache=None):
    """How many good diffs the parent would KEEP (not lose to the proposed
    child) under the forward matcher's rules (child wins ties).

    gate_cache optionally memoizes the parent-side scores, which are constant
    across one carve loop (only the child box changes between iterations).
    """
    goods = np.asarray(good_diffs, dtype=float)
    if goods.ndim == 1:
        goods = goods.reshape(1, -1)

    if gate_cache is not None and "p_score" in gate_cache:
        p_score = gate_cache["p_score"]
    else:
        p_score = bound_scores(goods, -parent_atol[0], parent_atol[1], parent_mask, softness)
        if gate_cache is not None:
            gate_cache["p_score"] = p_score

    c_score = bound_scores(goods, anchor_val - l_atol, anchor_val + h_atol, child_mask, softness)
    return int(np.sum(c_score < p_score))


def calculate_anchor_and_atol(family_mask, family_bads, remaining_bads,
                              selected_features, num_features, parent_atol,
                              parent_mask=None, gate_evidence_diffs=None,
                              max_keep=1, softness=0.1, gate_cache=None):
    """Pick the child cluster's anchor and tolerances from the family box.

    Anchor: the family row closest (L1) to the family median, skipping rows
    whose diff is all-zero (a zero diff is the parent's own anchor — it can't
    seed a distinct child). Tolerances per selected feature: the family's
    extent around the anchor plus ATOL_FLOOR, clamped so the child's box
    never reaches past the parent's edge (on features the parent constrains).

    Eat-parent gate: if the proposed box would leave the parent fewer than
    `max_keep` of its good diffs, collapse the tolerances to ATOL_FLOOR — the
    child then captures only (near-)exact matches of its anchor instead of
    swallowing the parent's population.

    Returns (anchor_idx_into_remaining_bads, atol (2, D), mask (D,)) or
    (None, None, None) when no valid anchor exists.
    """
    if len(family_bads) == 0:
        return None, None, None
    family_nonzero = np.any(np.abs(family_bads) > EPS, axis=1)
    if not np.any(family_nonzero):
        return None, None, None

    family_rows = np.where(family_mask)[0]
    nonzero_positions = np.where(family_nonzero)[0]
    candidates = family_bads[family_nonzero]
    median = np.median(candidates, axis=0)
    distances = np.sum(np.abs(candidates - median), axis=1)
    anchor_row = family_rows[nonzero_positions[int(np.argmin(distances))]]
    anchor_val = remaining_bads[anchor_row]

    mask = np.zeros(num_features, dtype=bool)
    mask[selected_features] = True

    family_min = np.min(family_bads, axis=0)
    family_max = np.max(family_bads, axis=0)

    l_atol = np.zeros(num_features)
    h_atol = np.zeros(num_features)
    for f in np.where(mask)[0]:
        val = anchor_val[f]
        fam_low = (val - family_min[f]) + ATOL_FLOOR
        fam_high = (family_max[f] - val) + ATOL_FLOOR

        # Clamp to the parent's edge, but only when the anchor genuinely sits
        # inside the parent's box on this feature (a non-positive distance to
        # the edge means the parent doesn't constrain this side).
        if parent_atol is not None and (parent_mask is None or bool(parent_mask[f])):
            par_low = val + parent_atol[0, f]
            par_high = parent_atol[1, f] - val
            if par_low <= 0:
                par_low = np.inf
            if par_high <= 0:
                par_high = np.inf
        else:
            par_low = par_high = np.inf

        l_atol[f] = max(min(fam_low, par_low), ATOL_FLOOR)
        h_atol[f] = max(min(fam_high, par_high), ATOL_FLOOR)

    if gate_evidence_diffs is not None and len(gate_evidence_diffs) > 0 and parent_atol is not None:
        pmask = parent_mask if parent_mask is not None else np.ones(num_features, dtype=bool)
        parent_keeps = count_parent_keeps(
            gate_evidence_diffs, anchor_val, l_atol, h_atol, mask,
            parent_atol, pmask, softness=softness, gate_cache=gate_cache)
        if parent_keeps < max_keep:
            l_atol = np.where(mask, ATOL_FLOOR, 0.0)
            h_atol = np.where(mask, ATOL_FLOOR, 0.0)

    return anchor_row, np.array([l_atol, h_atol]), mask


def carve(bad_diffs, good_diffs, remaining_indices, parent_atol, parent_mask,
          gate_cache=None, scorer=None):
    """One carve attempt: propose a child cluster for the remaining bads.

    bad_diffs / good_diffs: (B, D) / (G, D) diff-space rows for the parent's
    misrouted / correctly-routed samples. remaining_indices: which bad rows
    are still uncaptured by earlier carve iterations.

    Returns (anchor_idx, atol, mask, B, G, family_mask) — anchor_idx indexes
    into bad_diffs — or None when no box with positive net exists.
    """
    num_features = bad_diffs.shape[1]
    num_goods = len(good_diffs)
    remaining_bads = bad_diffs[remaining_indices]

    feature_results = score_all_features(remaining_bads, good_diffs, num_goods)

    # Don't pre-filter on per-feature net: even when every individual feature
    # scores net <= 0, AND-ing ranges can still reach net > 0 (e.g. a single
    # bad whose values each tie a different good on different features).
    if scorer is None:
        orderings = [feature_results]
    else:
        orderings = scorer.candidate_orderings(feature_results)

    best = None
    best_quality = None
    for ordering in orderings:
        candidate = build_family_intersection(
            ordering, remaining_bads, good_diffs, num_goods)
        quality = family_quality(candidate)
        if best is None or quality > best_quality:
            best, best_quality = candidate, quality

    family_size, family_mask, selected_features, _, final_G, net = best
    if family_size > 0 and net <= 0:
        return None  # the best box still traps as many goods as bads

    family_bads = remaining_bads[family_mask]
    anchor_idx, atol, mask = calculate_anchor_and_atol(
        family_mask, family_bads, remaining_bads,
        selected_features, num_features, parent_atol,
        parent_mask=parent_mask,
        gate_evidence_diffs=good_diffs,
        gate_cache=gate_cache,
    )
    if anchor_idx is None:
        return None

    # Map the anchor from remaining_bads back to the full bad_diffs row.
    return remaining_indices[anchor_idx], atol, mask, family_size, final_G, family_mask


def expand_atol_to_cover(current_atol, diffs):
    """Grow a cluster's (2, D) tolerance just enough to cover the given signed
    diffs. Infinite (never-constrained) entries collapse to exactly the
    required width; finite entries only ever grow."""
    if len(diffs) == 0:
        return current_atol

    diffs_arr = np.array(diffs)
    min_signed = np.min(diffs_arr, axis=0)
    max_signed = np.max(diffs_arr, axis=0)

    required_lower = np.zeros_like(min_signed)
    neg = min_signed < -EPS
    required_lower[neg] = np.abs(min_signed[neg])

    required_upper = np.zeros_like(max_signed)
    pos = max_signed > EPS
    required_upper[pos] = max_signed[pos]

    new_atol = current_atol.copy()
    for side, required in ((0, required_lower), (1, required_upper)):
        was_inf = np.isinf(new_atol[side])
        new_atol[side, was_inf] = required[was_inf]
        new_atol[side, ~was_inf] = np.maximum(new_atol[side, ~was_inf], required[~was_inf])
    return new_atol
