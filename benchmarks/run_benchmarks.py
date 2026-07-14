"""Honest benchmark: HypothesisTreeClassifier vs standard sklearn baselines.

5-fold stratified CV accuracy and ROC-AUC on small public datasets (sklearn
built-ins — no downloads). Prints a markdown table ready to paste into the
README. Expectation setting: the point of this table is honesty, not victory —
gradient boosting is expected to win on most tabular data.

    python benchmarks\\run_benchmarks.py
"""
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.datasets import load_breast_cancer, load_iris, load_wine, make_moons
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from hypothesis_tree import HypothesisTreeClassifier

RANDOM_STATE = 42
N_FOLDS = 5


def datasets():
    X, y = make_moons(n_samples=400, noise=0.25, random_state=RANDOM_STATE)
    yield "moons (400x2)", X, y
    X, y = load_iris(return_X_y=True)
    yield "iris (150x4)", X, y
    X, y = load_wine(return_X_y=True)
    yield "wine (178x13)", X, y
    X, y = load_breast_cancer(return_X_y=True)
    yield "breast_cancer (569x30)", X, y


def models():
    return {
        "HypothesisTree": lambda: HypothesisTreeClassifier(),
        "DecisionTree": lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE),
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
            random_state=RANDOM_STATE),
    }


def cv_scores(make_model, X, y):
    """Mean (accuracy, auc, fit_seconds) over stratified folds."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    accs, aucs, fit_times = [], [], []
    for train_idx, test_idx in skf.split(X, y):
        model = make_model()
        t0 = time.perf_counter()
        model.fit(X[train_idx], y[train_idx])
        fit_times.append(time.perf_counter() - t0)

        pred = model.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], pred))

        proba = model.predict_proba(X[test_idx])
        if len(np.unique(y)) == 2:
            aucs.append(roc_auc_score(y[test_idx], proba[:, 1]))
        else:
            aucs.append(roc_auc_score(y[test_idx], proba, multi_class="ovr"))
    return float(np.mean(accs)), float(np.mean(aucs)), float(np.mean(fit_times))


def main():
    warnings.filterwarnings("ignore")
    all_models = models()

    print(f"5-fold stratified CV (random_state={RANDOM_STATE}); "
          f"mean accuracy / mean ROC-AUC (OVR for multiclass) / mean fit time\n")
    header = "| Dataset | " + " | ".join(all_models) + " |"
    print(header)
    print("|" + "---|" * (len(all_models) + 1))

    for name, X, y in datasets():
        cells = []
        for make_model in all_models.values():
            acc, auc, fit_s = cv_scores(make_model, X, y)
            cells.append(f"{acc:.3f} / {auc:.3f} / {fit_s * 1000:.0f}ms")
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
