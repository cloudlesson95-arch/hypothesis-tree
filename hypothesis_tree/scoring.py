"""Feature scoring: which features (and value ranges) separate the errors.

Everything here works in *diff space*: each row is a misrouted ("bad") or
correctly routed ("good") sample expressed as its offset from the parent
cluster's anchor. The job is to find a box — a set of features, each with a
value range — that captures many bads and few goods. That box becomes a new
child cluster.

Two stages:

1. Per-feature sweep (`score_all_features`): for each feature, line up the
   bad and good values on a number line and run Kadane's max-subarray on the
   running (bads - goods) count. The best subarray is the value range where
   bads most outnumber goods; its score is `net = B - G`.

2. Family intersection (`build_family_intersection`): greedily AND feature
   ranges together, starting from the best-net feature, keeping each
   candidate feature only if the intersected box's net does not drop.

A *feature scorer* may propose alternative feature orderings for stage 2
(see `KadaneScorer`); the carve keeps whichever ordering yields the best box.
"""
from typing import NamedTuple

import numpy as np

# Range-membership tolerance: a value within EPS of a bound counts as inside.
EPS = 1e-9


class FeatureResult(NamedTuple):
    """Per-feature Kadane result used to rank features and seed the family."""
    net: float        # B - G over the kept range (higher = better separator)
    val_low: float    # kept range, extended toward the nearest good values
    val_high: float
    B: int            # bads inside the range
    G: int            # goods inside the range
    purity: float     # B / (B + G + 1)
    feature: int


def build_feature_scores(bad_vals, good_vals, goods_range=None):
    """Line up one feature's bad/good values for the Kadane sweep.

    Only goods inside the bads' value range matter (a good far outside any
    candidate range can never be trapped by it). goods_range optionally
    substitutes a different reference range (used by the family stage, where
    the bads shrink each iteration but the goods filter must stay stable).

    Returns (sorted_vals, b_counts, g_counts, scores) where scores = b - g
    per unique value.
    """
    if len(bad_vals) == 0:
        empty_i = np.array([], dtype=int)
        return np.array([]), empty_i, empty_i, np.array([], dtype=float)

    range_ref = goods_range if goods_range is not None else bad_vals
    lo, hi = np.min(range_ref), np.max(range_ref)

    if len(good_vals) > 0:
        in_range = (good_vals >= lo - EPS) & (good_vals <= hi + EPS)
        good_vals = good_vals[in_range]

    if len(good_vals) > 0:
        merged = np.concatenate([bad_vals, good_vals])
        sorted_vals, positions = np.unique(merged, return_inverse=True)
        b_counts = np.bincount(positions[:len(bad_vals)], minlength=len(sorted_vals))
        g_counts = np.bincount(positions[len(bad_vals):], minlength=len(sorted_vals))
    else:
        sorted_vals, positions = np.unique(bad_vals, return_inverse=True)
        b_counts = np.bincount(positions, minlength=len(sorted_vals))
        g_counts = np.zeros(len(sorted_vals), dtype=int)

    b_counts = b_counts.astype(int)
    g_counts = g_counts.astype(int)
    return sorted_vals, b_counts, g_counts, b_counts - g_counts


def kadane(scores, b_counts, g_counts):
    """Kadane's max-subarray over per-value scores, tracking the kept range's
    endpoints and B/G totals. The kept subarray must contain at least one bad.

    Returns (best_sum, start_idx, end_idx, B, G); zeros when nothing qualifies.
    """
    n = len(scores)
    if n == 0:
        return 0.0, 0, 0, 0, 0

    best_sum = float("-inf")
    best_start = best_end = 0
    best_B = best_G = 0

    curr_sum = 0.0
    curr_start = 0
    curr_B = curr_G = 0

    for i in range(n):
        curr_sum += scores[i]
        curr_B += b_counts[i]
        curr_G += g_counts[i]

        if curr_sum > best_sum and curr_B > 0:
            best_sum, best_start, best_end = curr_sum, curr_start, i
            best_B, best_G = curr_B, curr_G

        if curr_sum < 0:
            curr_sum = 0
            curr_start = i + 1
            curr_B = curr_G = 0

    if best_sum == float("-inf"):
        return 0.0, 0, 0, 0, 0
    return float(best_sum), best_start, best_end, int(best_B), int(best_G)


def extend_to_gap_bounds(sorted_vals, g_counts, start_idx, end_idx):
    """Widen Kadane's range up to (but not including) the nearest good value.

    Bads between the Kadane boundary and the nearest good get captured for
    free; the fence good itself stays one value outside the box, so it will
    still be out-of-bounds for the resulting cluster. A boundary value that
    already carries a good is left as-is.
    """
    val_low = sorted_vals[start_idx]
    if g_counts[start_idx] == 0:
        for i in range(start_idx - 1, -1, -1):
            if g_counts[i] > 0:
                val_low = sorted_vals[i + 1]
                break

    val_high = sorted_vals[end_idx]
    if g_counts[end_idx] == 0:
        for i in range(end_idx + 1, len(sorted_vals)):
            if g_counts[i] > 0:
                val_high = sorted_vals[i - 1]
                break

    return val_low, val_high


def score_all_features(remaining_bads, good_diffs, num_goods):
    """Kadane-sweep every feature; return FeatureResults sorted by
    (net desc, purity desc). The first entry seeds the family intersection."""
    results = []
    for f in range(remaining_bads.shape[1]):
        good_col = good_diffs[:, f] if num_goods > 0 else np.array([])
        sorted_vals, b_counts, g_counts, scores = build_feature_scores(
            remaining_bads[:, f], good_col)
        net, start_idx, end_idx, B, G = kadane(scores, b_counts, g_counts)

        if len(sorted_vals) > 0 and B > 0:
            val_low, val_high = extend_to_gap_bounds(sorted_vals, g_counts, start_idx, end_idx)
        else:
            val_low, val_high = 0.0, 0.0

        purity = B / (B + G + 1)
        results.append(FeatureResult(net, val_low, val_high, B, G, purity, f))

    results.sort(key=lambda r: (r.net, r.purity), reverse=True)
    return results


def build_family_intersection(feature_results, remaining_bads, good_diffs, num_goods):
    """Greedily AND feature ranges into a multi-feature box ("family").

    Seeded by the first (best-net) feature; each later candidate re-runs
    Kadane on the current family's values for that feature and is kept only
    if the intersected box's net does not drop below the running net.

    Returns (family_size, family_mask, selected_features, feature_ranges,
    final_G, net): the box's bad count, membership mask over remaining_bads,
    chosen features with their value ranges, trapped-good count, and net.
    """
    seed = feature_results[0]
    seed_f = seed.feature

    family_mask = ((remaining_bads[:, seed_f] >= seed.val_low - EPS)
                   & (remaining_bads[:, seed_f] <= seed.val_high + EPS))
    family_bads = remaining_bads[family_mask]
    family_size = len(family_bads)
    final_G = seed.G

    selected_features = [seed_f]
    feature_ranges = {seed_f: (seed.val_low, seed.val_high)}
    current_net = seed.net

    # Track which goods the committed ranges trap (the running AND). Each
    # candidate then range-checks only this shrinking subset.
    if num_goods > 0:
        trapped_idx = np.nonzero((good_diffs[:, seed_f] >= seed.val_low - EPS)
                                 & (good_diffs[:, seed_f] <= seed.val_high + EPS))[0]
    else:
        trapped_idx = None

    for result in feature_results[1:]:
        cand_f = result.feature

        good_col = good_diffs[:, cand_f] if num_goods > 0 else np.array([])
        sorted_vals, b_counts, g_counts, scores = build_feature_scores(
            family_bads[:, cand_f], good_col, goods_range=remaining_bads[:, cand_f])
        _, start_idx, end_idx, cand_B, _ = kadane(scores, b_counts, g_counts)

        if len(sorted_vals) == 0 or cand_B == 0:
            continue

        cand_low, cand_high = extend_to_gap_bounds(sorted_vals, g_counts, start_idx, end_idx)

        cand_range_mask = ((remaining_bads[:, cand_f] >= cand_low - EPS)
                           & (remaining_bads[:, cand_f] <= cand_high + EPS))
        intersection_mask = family_mask & cand_range_mask
        new_B = int(np.sum(intersection_mask))

        if trapped_idx is not None and new_B > 0:
            trapped_vals = good_diffs[trapped_idx, cand_f]
            cand_traps = (trapped_vals >= cand_low - EPS) & (trapped_vals <= cand_high + EPS)
            new_G = int(np.sum(cand_traps))
        else:
            cand_traps = None
            new_G = 0

        new_net = new_B - new_G
        if new_net >= current_net:
            family_mask = intersection_mask
            family_bads = remaining_bads[family_mask]
            family_size = new_B
            final_G = new_G
            selected_features.append(cand_f)
            feature_ranges[cand_f] = (cand_low, cand_high)
            if trapped_idx is not None and cand_traps is not None:
                trapped_idx = trapped_idx[cand_traps]
            current_net = new_net

    return family_size, family_mask, selected_features, feature_ranges, final_G, current_net


def family_quality(family):
    """(net, purity) sort key for picking the best of several candidate
    families built from different feature orderings."""
    family_size, _, _, _, final_G, net = family
    purity = family_size / (family_size + final_G + 1)
    return net, purity


# ── Feature scorer seam ───────────────────────────────────────────────────
# A scorer proposes one or more candidate feature ORDERINGS for the family
# intersection; the carve builds a family per ordering and keeps the best.
# The built-in Kadane scorer proposes exactly the Kadane-net ordering, i.e.
# the baseline behavior. To plug in your own (e.g. a learned importance
# ranking), pass any object with a `candidate_orderings(feature_results)`
# method — or a factory `f(bad_diffs, good_diffs) -> scorer` — as the
# classifier's `feature_scorer`.


class KadaneScorer:
    """Default scorer: the single Kadane-net ordering, unchanged."""

    def __init__(self, bad_diffs=None, good_diffs=None):
        pass

    def candidate_orderings(self, feature_results):
        """One or more feature orderings for the family stage to try."""
        return [feature_results]


FEATURE_SCORERS = {"kadane": KadaneScorer}


def make_scorer(feature_scorer, bad_diffs, good_diffs):
    """Resolve the classifier's `feature_scorer` argument into a scorer
    instance: a registry name ("kadane") or a factory callable."""
    if isinstance(feature_scorer, str):
        try:
            factory = FEATURE_SCORERS[feature_scorer]
        except KeyError:
            raise ValueError(
                f"unknown feature_scorer {feature_scorer!r}; "
                f"expected one of {sorted(FEATURE_SCORERS)} or a callable"
            ) from None
        return factory(bad_diffs, good_diffs)
    return feature_scorer(bad_diffs, good_diffs)
