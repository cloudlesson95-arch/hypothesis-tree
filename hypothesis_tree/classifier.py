"""Scikit-learn-compatible wrapper around the hypothesis tree."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array

from .refinement import run_backward
from .tree import HypothesisTree


class HypothesisTreeClassifier(BaseEstimator, ClassifierMixin):
    """An exemplar-based tree classifier grown by error-driven carving.

    The model is a tree of prototype clusters: each cluster stores an actual
    training row (its anchor), an active-feature mask, a per-feature
    tolerance box, and a class label. Prediction routes a sample down the
    tree through parent-vs-children match competitions and returns the label
    of the deepest matched cluster — so every prediction is traceable to one
    stored exemplar and its box.

    Training is epoch-based: route the training set, then for each cluster
    that caught misclassified samples, carve child clusters around the value
    ranges where errors concentrate (per-feature Kadane scan + greedy
    multi-feature intersection). The tree only grows where it makes mistakes.

    Parameters
    ----------
    max_epochs : int, default=10
        Maximum forward/backward epochs over the training set.
    early_stop_epochs : int, default=3
        Stop after this many consecutive epochs with zero training errors.
    carve_budget : int, default=5000
        Hard cap on carve iterations per erroring cluster per epoch.
    carve_budget_step : int, default=200
        Per-epoch ramp of the carve cap: epoch e allows
        min(carve_budget, carve_budget_step * (e + 1)) iterations, so early
        epochs carve conservatively and later epochs may finish the job. Set
        carve_budget_step >= carve_budget to disable the ramp.
    softness : float, default=0.1
        Decay scale of the out-of-box match score, exp(-distance/softness),
        for the "scored" strategy. Note that within one competition the
        ranking is monotonic in distance, so this rarely changes routing.
    match_strategy : str or callable, default="scored"
        Forward matcher: "scored" (nearest box via elastic-Gaussian score) or
        "closer" (per-feature closest-candidate vote ratio, tolerance-gated).
        A callable with `batched_match_scored`'s signature also works.
    feature_scorer : str or callable, default="kadane"
        Which feature-ordering scorer the carve uses. "kadane" (the only
        built-in) ranks features by the Kadane net (bads - goods). Extension
        point: pass a factory `f(bad_diffs, good_diffs) -> scorer` where the
        scorer has `candidate_orderings(feature_results)`.
    verbose : int, default=0
        1 prints a one-line summary per epoch.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Sorted class labels seen in fit.
    tree_ : HypothesisTree
        The fitted cluster tree (inspect it for interpretability).
    n_epochs_ : int
        Epochs actually run.

    Notes
    -----
    Fitting is deterministic but order-dependent: the first training row
    seeds the root cluster. Shuffle consistently if you need reproducible
    comparisons across preprocessing changes.
    """

    def __init__(self, max_epochs=10, early_stop_epochs=3, carve_budget=5000,
                 carve_budget_step=200, softness=0.1, match_strategy="scored",
                 feature_scorer="kadane", verbose=0):
        self.max_epochs = max_epochs
        self.early_stop_epochs = early_stop_epochs
        self.carve_budget = carve_budget
        self.carve_budget_step = carve_budget_step
        self.softness = softness
        self.match_strategy = match_strategy
        self.feature_scorer = feature_scorer
        self.verbose = verbose

    def fit(self, X, y):
        """Grow the tree on (X, y): route, carve where routing erred, repeat
        until a zero-error epoch (with patience) or the epoch budget."""
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]

        tree = HypothesisTree(softness=self.softness,
                              match_strategy=self.match_strategy)
        tree.seed_root(X[0], y[0])

        consecutive_perfect = 0
        epoch = -1
        for epoch in range(self.max_epochs):
            pred, _ = tree.forward(X)
            n_errors = int(np.sum(pred != y))

            carve_cap = min(self.carve_budget,
                            self.carve_budget_step * (epoch + 1))
            created = run_backward(tree, y, pred, max_iterations=carve_cap,
                                   feature_scorer=self.feature_scorer)

            if self.verbose:
                print(f"epoch {epoch}: errors={n_errors} "
                      f"new_clusters={created} total_clusters={len(tree)}")

            if n_errors == 0:
                consecutive_perfect += 1
                if consecutive_perfect >= self.early_stop_epochs:
                    break
            else:
                consecutive_perfect = 0

        self.tree_ = tree
        self.n_epochs_ = epoch + 1
        return self

    def predict(self, X):
        """Route each row through the fitted tree and return the label of the
        deepest matched cluster."""
        check_is_fitted(self)
        X = check_array(X)
        pred, _ = self.tree_.forward(X)
        return pred

    def predict_proba(self, X):
        """Confidence-based probabilities: the predicted class receives the
        matched cluster's confidence; the remaining mass spreads uniformly
        over the other classes."""
        check_is_fitted(self)
        X = check_array(X)
        pred, conf = self.tree_.forward(X)

        n_classes = len(self.classes_)
        class_to_col = {c: i for i, c in enumerate(self.classes_)}
        conf = np.asarray(conf, dtype=float)

        if n_classes == 1:
            return np.ones((len(pred), 1))

        off_class = (1.0 - conf) / (n_classes - 1)
        probas = np.repeat(off_class[:, None], n_classes, axis=1)
        cols = np.fromiter((class_to_col.get(p, -1) for p in pred),
                           dtype=np.int64, count=len(pred))
        known = cols >= 0
        probas[np.nonzero(known)[0], cols[known]] = conf[known]
        return probas
