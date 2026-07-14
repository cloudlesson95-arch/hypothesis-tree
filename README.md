# hypothesis-tree

Yes, it's another tree classifier. The industry definitely needed one more.

An **exemplar-based tree classifier** for tabular data, grown by error-driven
carving. Every cluster in the tree is an actual training row (an exemplar)
plus a per-feature tolerance box and a class label — so every prediction is
traceable: *"this sample routed to the cluster anchored on that training row,
matching on these features, within these bounds."*

What it offers:

- **Traceable predictions.** No weights, no ensembles of hundreds of trees:
  a prediction is one descent path ending at one stored exemplar you can
  print and inspect.
- **Grows only where it errs.** The model starts as a single cluster and adds
  a cluster only when a group of misclassified samples justifies one, with an
  explicit bads-vs-goods net score for every carve.
- **Small and dependency-light.** Pure numpy + scikit-learn API, ~900 lines,
  written to be read.

Note, that the basic version will not beat gradient boosting on accuracy (see the honest benchmarks below). 

## Install

```
git clone <this repo>
cd hypothesis-tree
pip install -e .
```

Dependencies: `numpy`, `scikit-learn` (Python ≥ 3.10).

## Quickstart

```python
from hypothesis_tree import HypothesisTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

clf = HypothesisTreeClassifier(verbose=1).fit(X_train, y_train)
print((clf.predict(X_test) == y_test).mean())

# Trace one prediction back to its exemplar:
tree = clf.tree_
pred, conf = tree.forward(X_test[0])
path = tree.history[0]["path"]
print(f"class {pred[0]} via clusters {path}; exemplar = {tree.X[path[-1]]}")
```

`examples/quickstart.py` is the runnable version.

## How it works

1. **A cluster is a box.** Anchor row + active-feature mask + asymmetric
   per-feature tolerance (how far below/above the anchor still counts as
   inside) + a class label. The first training row seeds a root with infinite
   tolerance — the untrained model predicts one class for everything.

2. **Forward = routing.** A sample descends the tree through
   parent-vs-children match competitions. Matching scores 1.0 inside a box
   and decays as `exp(-distance/softness)` outside it; a child that at least
   ties its parent captures the sample. The deepest matched cluster's label
   is the prediction ([matching.py](hypothesis_tree/matching.py),
   [tree.py](hypothesis_tree/tree.py)).

3. **Backward = carving.** After each epoch, misclassified samples are
   grouped by the cluster that caught them. For each such cluster, a
   per-feature **Kadane max-subarray scan** finds the value range where
   errors most outnumber correct samples; ranges are greedily ANDed across
   features into a multi-feature box with positive net (bads − goods). The
   box becomes a child cluster labeled with the errors' true class
   ([scoring.py](hypothesis_tree/scoring.py),
   [carving.py](hypothesis_tree/carving.py),
   [refinement.py](hypothesis_tree/refinement.py)).

4. **Guard rails.** Boxes are clamped to their parent's bounds; an
   *eat-parent gate* simulates routing and collapses any box that would steal
   essentially all of its parent's correctly-routed samples; boxes that trap
   more goods than bads become "ambiguity carves" that keep the parent's
   label with reduced confidence.

Training repeats forward + backward until an epoch has zero errors (with
patience) or the epoch budget runs out.

For a fully worked example — one carve traced end to end with real numbers —
see [docs/how_it_works.md](docs/how_it_works.md).

## Benchmarks (honest ones)

5-fold stratified CV, default parameters everywhere, cells are
**accuracy / ROC-AUC (OVR) / mean fit time**. Reproduce with
`python benchmarks\run_benchmarks.py`.

| Dataset | HypothesisTree | DecisionTree | RandomForest | HistGradientBoosting |
|---|---|---|---|---|
| moons (400x2) | 0.843 / 0.842 / 20ms | 0.890 / 0.890 / 1ms | 0.920 / 0.958 / 72ms | 0.915 / 0.961 / 354ms |
| iris (150x4) | 0.933 / 0.950 / 11ms | 0.953 / 0.965 / 1ms | 0.947 / 0.994 / 62ms | 0.940 / 0.986 / 81ms |
| wine (178x13) | 0.826 / 0.866 / 21ms | 0.893 / 0.919 / 1ms | 0.977 / 0.999 / 67ms | 0.966 / 0.998 / 93ms |
| breast_cancer (569x30) | 0.902 / 0.882 / 91ms | 0.910 / 0.900 / 6ms | 0.956 / 0.989 / 118ms | 0.958 / 0.991 / 128ms |

Reading guide: accuracy lands in single-decision-tree territory; the
ensembles win, as they do against nearly everything on tabular data. The
AUC gap is structural — probabilities come from per-cluster confidence, not
from averaging many trees. If you need the last few points of accuracy, use
boosting; if you need to point at the exact training example behind a
prediction, that's what this is for.

Tip — feature scale matters: boxes live in raw feature units, so one
wide-range feature can dominate the match distance. Standardizing lifts wine
to 0.893 / 0.917 (single-tree parity, +6.7 accuracy points) while leaving
iris unchanged and moons / breast_cancer marginally lower — if your features
span very different ranges, put a `StandardScaler` in front.

### OpenML suite

The same protocol over 14 OpenML datasets (all-numeric, no missing values,
pinned by `data_id`). Reproduce with `python benchmarks\run_openml.py` —
datasets download once into sklearn's cache.

| Dataset | HypothesisTree | DecisionTree | RandomForest | HistGradientBoosting |
|---|---|---|---|---|
| banknote (1372x4) | 0.926 / 0.927 / 97ms | 0.983 / 0.983 / 2ms | 0.993 / 1.000 / 123ms | 0.994 / 1.000 / 442ms |
| blood-transfusion (748x4) | 0.762 / 0.532 / 54ms | 0.710 / 0.573 / 1ms | 0.749 / 0.686 / 84ms | 0.749 / 0.691 / 195ms |
| diabetes (768x8) | 0.706 / 0.667 / 505ms | 0.700 / 0.672 / 5ms | 0.769 / 0.824 / 293ms | 0.746 / 0.799 / 592ms |
| ionosphere (351x34) | 0.892 / 0.895 / 226ms | 0.897 / 0.889 / 11ms | 0.934 / 0.978 / 290ms | 0.943 / 0.968 / 349ms |
| sonar (208x60) | 0.669 / 0.658 / 352ms | 0.712 / 0.712 / 8ms | 0.827 / 0.927 / 267ms | 0.841 / 0.935 / 207ms |
| vehicle (846x18) | 0.609 / 0.741 / 1.0s | 0.692 / 0.795 / 10ms | 0.733 / 0.929 / 340ms | 0.771 / 0.928 / 2.4s |
| qsar-biodeg (1055x41) | 0.808 / 0.743 / 1.9s | 0.817 / 0.797 / 22ms | 0.871 / 0.935 / 455ms | 0.882 / 0.936 / 892ms |
| kc1 (2109x21) | 0.850 / 0.594 / 907ms | 0.814 / 0.609 / 9ms | 0.861 / 0.825 / 188ms | 0.857 / 0.776 / 164ms |
| pc1 (1109x21) | 0.924 / 0.610 / 284ms | 0.910 / 0.675 / 5ms | 0.937 / 0.848 / 127ms | 0.930 / 0.833 / 149ms |
| steel-plates-fault (1941x33) | 0.853 / 0.792 / 1.6s | 1.000 / 1.000 / 9ms | 0.993 / 1.000 / 237ms | 1.000 / 1.000 / 91ms |
| climate-crashes (540x20) | 0.896 / 0.510 / 126ms | 0.881 / 0.620 / 4ms | 0.917 / 0.813 / 106ms | 0.906 / 0.844 / 101ms |
| segment (2310x18) | 0.913 / 0.951 / 350ms | 0.956 / 0.974 / 11ms | 0.972 / 0.998 / 240ms | 0.980 / 0.999 / 755ms |
| wilt (4839x5) | 0.938 / 0.530 / 2.2s | 0.977 / 0.885 / 9ms | 0.982 / 0.989 / 349ms | 0.984 / 0.986 / 166ms |
| phoneme (5404x5) | 0.775 / 0.715 / 3.7s | 0.872 / 0.843 / 18ms | 0.910 / 0.961 / 570ms | 0.896 / 0.952 / 156ms |

Mean accuracy rank (1 = best): HistGradientBoosting 1.46, RandomForest 1.79,
DecisionTree 3.25, HypothesisTree 3.50.

The wider suite sharpens the honest picture. Accuracy stays in
single-decision-tree territory overall (it beats the tree outright on 5 of
14 — e.g. blood-transfusion, where it is best of all four models). Two
weaknesses show clearly: on **class-imbalanced** datasets (blood-transfusion,
kc1, pc1, climate-crashes, wilt) accuracy holds up but AUC collapses toward
0.5–0.6 — the coarse confidence-based probabilities carry little ranking
signal — and axis-aligned single-value structure (steel-plates-fault, where
one feature is essentially the label) suits classic trees far better. Fit
time grows with error count, reaching a few seconds on the 5k-row sets.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `max_epochs` | 10 | forward/backward epochs over the training set |
| `early_stop_epochs` | 3 | stop after this many consecutive zero-error epochs |
| `carve_budget` | 5000 | max carve iterations per erroring cluster per epoch |
| `carve_budget_step` | 200 | per-epoch ramp of the carve cap (epoch e allows `step*(e+1)`) |
| `softness` | 0.1 | decay scale of the out-of-box match score (rarely changes routing: rankings are monotonic in distance) |
| `match_strategy` | `"scored"` | forward matcher: `"scored"` (nearest box) or `"closer"` (per-feature closest-candidate vote ratio; carves more clusters — better on wine, slightly worse on moons/iris in our CV) |
| `feature_scorer` | `"kadane"` | feature-ordering strategy for the carve; extension point — pass a factory `f(bad_diffs, good_diffs) -> scorer` with `candidate_orderings()` |
| `verbose` | 0 | 1 prints a per-epoch summary |

## Project layout

```
hypothesis_tree/
  classifier.py   sklearn-compatible wrapper (fit/predict/predict_proba)
  tree.py         cluster memory + batched routing forward pass
  matching.py     elastic-Gaussian box matcher + descent tie-breaks
  refinement.py   backward pass: gather errors, order clusters, carve loop
  scoring.py      per-feature Kadane sweep + greedy family intersection
  carving.py      anchor/tolerance selection, eat-parent gate
benchmarks/       CV comparison vs sklearn baselines (table above)
docs/             how it works — a worked carve example, end to end
examples/         quickstart
tests/            unit tests (pytest, run in CI on Python 3.10-3.13)
```

## Status & roadmap

Alpha. The implementation is verified prediction-identical to the research
codebase it was distilled from (both matcher strategies, same data and
schedule). Possible next steps: a broader benchmark suite (OpenML), an
MLP-based feature scorer as a plug-in, and performance work for large
datasets (vectorized scans, match chunking, GPU backend).

Known limitations, stated up front:

- Not competitive with boosted ensembles on accuracy (see table).
- Fitting is order-dependent (the first training row seeds the root).
- Probabilities are coarse (cluster confidence, not calibrated).
- No missing-value or categorical handling — encode features numerically first.

## License

MIT.
