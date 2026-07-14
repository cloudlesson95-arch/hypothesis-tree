"""Backward pass: turn one epoch's routing errors into new child clusters.

After a forward pass over the training data, every misrouted sample ("bad")
sits at some leaf cluster whose label it contradicts; every correctly routed
sample ("good") is evidence for where boxes must NOT grow. The backward pass:

1. groups bads by the leaf that caught them (`gather_backward_data`),
2. widens the boxes of leaves that only saw goods (cheap, always safe),
3. visits the erroring leaves — most bads first — and repeatedly carves a
   child cluster around the densest still-uncaptured group of bads
   (`resolve_cluster`), until the bads run out or a budget/stagnation stop.

A carved child usually takes the bads' true label (it fixes them on the next
forward). When the proposed box would trap more goods than bads it instead
duplicates the parent's label as an "ambiguity carve", recording the impurity
in its confidence.
"""
from collections import defaultdict

import numpy as np

from .carving import carve, expand_atol_to_cover
from .scoring import make_scorer

# Consecutive carve iterations allowed to leave the remaining-bads count
# unchanged before the resolve loop bails out (defensive: a healthy carve
# always captures at least its own anchor).
STAGNATION_LIMIT = 5


def gather_backward_data(X, y, pred, history):
    """Regroup the forward pass's per-sample history for the backward pass.

    Returns (bad_inputs_by_cluster, node_stats, leaf_stats):
    - bad_inputs_by_cluster: leaf index -> list of bad-sample records
      (input_sample, target, path, leaf_diff);
    - node_stats: node index -> good/bad diffs seen at that node. A good
      counts only at its final stop (a good that merely passed through a
      node is not evidence that the node's box must contain it); a bad
      counts at every node along its path;
    - leaf_stats: leaf index -> good/bad diffs of samples that stopped there
      (drives the pure-good box widening).
    """
    bad_inputs_by_cluster = defaultdict(list)
    node_stats = defaultdict(lambda: {"good_diffs": [], "bad_diffs": []})
    leaf_stats = defaultdict(lambda: {"good_diffs": [], "bad_diffs": []})

    for i in range(len(X)):
        record = history[i]
        path = record["path"]
        input_diffs = record["input_diffs"]
        leaf_idx = path[-1] if path else -1
        is_good = pred[i] == y[i]

        if leaf_idx != -1 and input_diffs:
            leaf_diff = input_diffs[-1]
            side = "good_diffs" if is_good else "bad_diffs"
            leaf_stats[leaf_idx][side].append(leaf_diff)

        if not is_good and leaf_idx != -1:
            bad_inputs_by_cluster[leaf_idx].append({
                "input_sample": record["input_sample"],
                "target": y[i],
                "path": path,
                "leaf_diff": input_diffs[-1] if input_diffs else None,
            })

        for depth, node_idx in enumerate(path):
            if node_idx == -1 or depth >= len(input_diffs):
                continue
            if is_good:
                if node_idx == leaf_idx:
                    node_stats[node_idx]["good_diffs"].append(input_diffs[depth])
            else:
                node_stats[node_idx]["bad_diffs"].append(input_diffs[depth])

    return bad_inputs_by_cluster, node_stats, leaf_stats


def _leaf_diff_size(bad):
    """L1 size of a bad sample's leaf diff (its distance from the leaf's
    anchor); -1.0 when the diff is missing."""
    leaf_diff = bad["leaf_diff"]
    return np.sum(np.abs(leaf_diff)) if leaf_diff is not None else -1.0


def sort_clusters_by_count_and_diff(bad_inputs_by_cluster):
    """Refinement order: clusters with the most bads first, ties broken by the
    largest individual bad diff. Each cluster's bads are also sorted largest
    diff first, so the carve loop attacks the farthest outliers first."""
    for bads in bad_inputs_by_cluster.values():
        bads.sort(key=_leaf_diff_size, reverse=True)
    return sorted(
        bad_inputs_by_cluster.items(),
        key=lambda item: (len(item[1]), _leaf_diff_size(item[1][0])),
        reverse=True,
    )


def resolve_cluster(tree, parent_idx, bad_list, good_diffs, max_iterations,
                    feature_scorer):
    """Carve children off one erroring cluster until its bads are captured.

    Each iteration proposes one child box over the still-uncaptured bads
    (see carving.carve) and adds it to the tree. Stops on: all bads captured,
    no positive-net box left, the iteration budget, or stagnation.

    Returns the number of clusters created.
    """
    bads = [b for b in bad_list if b["leaf_diff"] is not None]
    if not bads:
        return 0

    bad_diffs = np.array([b["leaf_diff"] for b in bads])
    good_arr = np.array(good_diffs)
    parent_mask = tree.masks[parent_idx]

    scorer = make_scorer(feature_scorer, bad_diffs, good_arr)
    gate_cache = {}  # parent-side gate scores are constant across this loop

    remaining = list(range(len(bads)))
    created = 0
    iteration = 0
    stagnation = 0
    captured_bads = 0
    captured_goods = 0

    while remaining and iteration < max_iterations:
        iteration += 1

        result = carve(bad_diffs, good_arr, remaining,
                       parent_atol=tree.atol[parent_idx],
                       parent_mask=parent_mask,
                       gate_cache=gate_cache, scorer=scorer)
        if result is None:
            break
        anchor_idx, atol, mask, n_bads, n_goods, family_mask = result
        if anchor_idx >= len(bads):
            break  # defensive: anchor must map back to a known bad sample

        anchor = bads[anchor_idx]
        if n_bads > n_goods:
            # The box separates cleanly: the child takes the bads' label.
            # From the child's viewpoint the parent's bads are its goods.
            label = anchor["target"]
            child_bads, child_goods = n_goods, n_bads
        else:
            # Mostly goods trapped: duplicate the parent's label so those
            # samples keep predicting correctly ("ambiguity carve"); the
            # impurity lands in the child's confidence.
            label = tree.y[parent_idx]
            child_bads, child_goods = n_bads, n_goods

        total = n_bads + n_goods
        confidence = child_goods / total if total > 0 else 1.0
        tree.add_child(anchor["path"][-1], anchor["input_sample"], mask, atol,
                       label, confidence, child_bads, child_goods)
        created += 1
        captured_bads += int(np.sum(family_mask))
        captured_goods += n_goods

        new_remaining = [remaining[i] for i, captured in enumerate(family_mask)
                         if not captured]
        if len(new_remaining) >= len(remaining):
            stagnation += 1
            if stagnation >= STAGNATION_LIMIT:
                break
        else:
            stagnation = 0
        remaining = new_remaining

    if created > 0:
        # Re-estimate the parent's confidence from what the carves left it.
        b_left = max(0, len(bad_diffs) - captured_bads)
        g_left = max(0, len(good_diffs) - captured_goods)
        tree.b_count[parent_idx] = b_left
        tree.g_count[parent_idx] = g_left
        total = b_left + g_left
        tree.confidence[parent_idx] = g_left / total if total > 0 else 1.0

    return created


def run_backward(tree, y, pred, max_iterations, feature_scorer="kadane"):
    """One backward pass over the tree's last forward. Returns the number of
    clusters created."""
    if not np.any(pred != np.asarray(y)):
        return 0

    X = tree.last_inputs
    bad_inputs_by_cluster, node_stats, leaf_stats = gather_backward_data(
        X, y, pred, tree.history)

    # Leaves that saw only goods: widen their box to cover them, so borderline
    # samples that routed correctly stop being borderline.
    for cluster_idx, leaf in leaf_stats.items():
        if leaf["good_diffs"] and not leaf["bad_diffs"]:
            tree.atol[cluster_idx] = expand_atol_to_cover(
                tree.atol[cluster_idx], leaf["good_diffs"])

    created = 0
    for cluster_idx, bad_list in sort_clusters_by_count_and_diff(bad_inputs_by_cluster):
        good_diffs = node_stats[cluster_idx]["good_diffs"]
        # The parent must first own its goods (cover them in-box); the carve
        # then sizes children against that settled parent box.
        if good_diffs:
            tree.atol[cluster_idx] = expand_atol_to_cover(
                tree.atol[cluster_idx], good_diffs)

        created += resolve_cluster(tree, cluster_idx, bad_list, good_diffs,
                                   max_iterations, feature_scorer)
    return created
