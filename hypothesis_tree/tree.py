"""The hypothesis tree: cluster memory plus the routing forward pass.

A fitted tree is a set of clusters stored in parallel lists (one entry per
cluster) plus parent/child wiring. Each cluster is an exemplar row from the
training data, an active-feature mask, an asymmetric per-feature tolerance
box, and a class label — so every prediction traces back to "this sample
routed to the cluster anchored on that training row".

The forward pass routes all samples through the tree in lockstep: one root
competition, then repeated parent-vs-children matches, descending while a
child wins. The prediction is the label of the final (deepest) matched
cluster. Alongside predictions, `forward` records each sample's descent path
and per-node diffs — the raw material the backward pass consumes.
"""
from collections import defaultdict

import numpy as np

from .matching import get_match_fn


class HypothesisTree:
    """Cluster memory + tree wiring + batched routing."""

    def __init__(self, softness=0.1, match_strategy="scored"):
        self.softness = softness
        self.match_strategy = match_strategy
        self._match = get_match_fn(match_strategy)

        # One entry per cluster, index-aligned:
        self.X = []           # anchor row (exemplar)
        self.masks = []       # active-feature bool mask
        self.atol = []        # (2, D) tolerance: [how far below, how far above]
        self.y = []           # class label the cluster votes for
        self.confidence = []  # goods / (bads + goods) at creation time
        self.b_count = []     # bads captured at creation
        self.g_count = []     # goods captured at creation

        # Tree structure.
        self.roots = []
        self.children = defaultdict(list)  # parent index -> [child indices]
        self.parents = {}                  # child index -> parent index (-1 for roots)

        # Set by the last forward pass; consumed by the backward pass.
        self.last_inputs = None  # the routed rows (float32)
        self.history = []        # per-sample dicts: path, input_diffs, input_sample

    def __len__(self):
        return len(self.X)

    def add_cluster(self, x, mask, atol, label, confidence=1.0, b_count=0, g_count=0):
        """Append a cluster; the caller wires it into the tree."""
        idx = len(self.X)
        self.X.append(x)
        self.masks.append(mask)
        self.atol.append(atol)
        self.y.append(label)
        self.confidence.append(confidence)
        self.b_count.append(b_count)
        self.g_count.append(g_count)
        return idx

    def add_child(self, parent_idx, x, mask, atol, label, confidence, b_count, g_count):
        """Append a cluster and wire it as a child of `parent_idx`."""
        idx = self.add_cluster(x, mask, atol, label, confidence, b_count, g_count)
        self.children[parent_idx].append(idx)
        self.parents[idx] = parent_idx
        return idx

    def seed_root(self, x, label):
        """Create the root cluster: all features active, infinite tolerance —
        it matches everything, so the untrained tree predicts `label` for any
        input. Training carves the rest of the class structure under it."""
        x = np.asarray(x, dtype=np.float64)
        idx = self.add_cluster(
            x,
            np.ones_like(x, dtype=bool),
            np.full((2, len(x)), np.inf),
            label,
        )
        self.roots.append(idx)
        self.parents[idx] = -1

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, X):
        """Route rows through the tree. Returns (predictions, confidences)
        and records `last_inputs` / `history` for a subsequent backward pass.

        float32 throughout: it halves the match tensor's bandwidth and, since
        every stored tolerance is derived with a small floor (never exact),
        the routing decisions are insensitive to the precision drop.
        """
        x = np.asarray(X, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self.last_inputs = x

        mem_X = np.asarray(self.X, dtype=np.float32)
        mem_masks = np.asarray(self.masks)
        mem_atol = np.asarray(self.atol, dtype=np.float32)

        paths = self._route(x, mem_X, mem_masks, mem_atol)
        return self._finalize(x, paths, mem_X)

    def _route(self, x, mem_X, mem_masks, mem_atol):
        """Batched tree-walk. Returns one descent path (list of cluster
        indices, root first) per sample; empty list = no root matched."""
        n = x.shape[0]
        current_node = np.full(n, -1, dtype=int)
        done = np.zeros(n, dtype=bool)
        paths = [[] for _ in range(n)]

        # Root competition: every sample vs every root, one batched call.
        winners = self._match(x, mem_X, mem_masks, mem_atol, list(self.roots),
                              parent_pos=None, softness=self.softness)
        matched = winners != -1
        current_node[matched] = winners[matched]
        for i in np.where(matched)[0]:
            paths[i].append(int(winners[i]))
        done[~matched] = True

        # Descent: group still-active samples by their current node and let
        # each group's node compete against its children in one batched call.
        while True:
            active = np.where(~done)[0]
            if len(active) == 0:
                break
            nodes_here = current_node[active]
            for node_id in np.unique(nodes_here):
                node_id = int(node_id)
                samples_idx = active[nodes_here == node_id]
                children = self.children.get(node_id, [])
                if not children:
                    done[samples_idx] = True  # leaf reached
                    continue
                competitors = [node_id] + list(children)
                winners = self._match(
                    x[samples_idx], mem_X, mem_masks, mem_atol, competitors,
                    parent_pos=0, softness=self.softness)
                descended = (winners != node_id) & (winners != -1)
                done[samples_idx[~descended]] = True  # parent kept the sample
                for i, w in zip(samples_idx[descended], winners[descended]):
                    current_node[i] = w
                    paths[int(i)].append(int(w))
        return paths

    def _finalize(self, x, paths, mem_X):
        """Turn descent paths into predictions/confidences and record the
        per-sample history (path + diff from each node's anchor) that the
        backward pass reads."""
        n = x.shape[0]
        mem_y = np.asarray(self.y)
        mem_conf = np.asarray(self.confidence)

        leaf = np.fromiter((p[-1] if p else -1 for p in paths), dtype=np.int64, count=n)
        has_path = leaf >= 0

        predictions = np.zeros(n, dtype=mem_y.dtype)
        confidences = np.zeros(n, dtype=float)
        predictions[has_path] = mem_y[leaf[has_path]]
        confidences[has_path] = mem_conf[leaf[has_path]]

        self.history = [
            {
                "path": paths[i],
                "input_diffs": [x[i] - mem_X[node] for node in paths[i]],
                "input_sample": x[i],
            }
            for i in range(n)
        ]
        return predictions, confidences
