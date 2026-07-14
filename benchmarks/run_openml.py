"""OpenML benchmark suite: HypothesisTreeClassifier vs sklearn baselines.

Same protocol as run_benchmarks.py (5-fold stratified CV, accuracy /
ROC-AUC / mean fit time), over a fixed suite of small-to-medium OpenML
datasets — all-numeric features, no missing values, pinned by data_id so the
table is reproducible. Datasets are downloaded once and cached by sklearn
(~/scikit_learn_data).

    python benchmarks\\run_openml.py            # full suite
    python benchmarks\\run_openml.py sonar kc1  # subset, by name

Expectation setting, as in the README: the point is honesty, not victory —
gradient boosting is expected to win on most tabular data.
"""
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from hypothesis_tree import HypothesisTreeClassifier

RANDOM_STATE = 42
N_FOLDS = 5

# name -> OpenML data_id (pinned; all-numeric features, no missing values).
SUITE = {
    "banknote": 1462,        # 1372 x 4, binary
    "blood-transfusion": 1464,  # 748 x 4, binary
    "diabetes": 37,          # 768 x 8, binary
    "ionosphere": 59,        # 351 x 34, binary
    "sonar": 40,             # 208 x 60, binary
    "vehicle": 54,           # 846 x 18, 4 classes
    "qsar-biodeg": 1494,     # 1055 x 41, binary
    "kc1": 1067,             # 2109 x 21, binary
    "pc1": 1068,             # 1109 x 21, binary
    "steel-plates-fault": 1504,  # 1941 x 33, binary
    "climate-crashes": 1467,  # 540 x 20, binary
    "segment": 40984,        # 2310 x 16, 7 classes
    "wilt": 40983,           # 4839 x 5, binary
    "phoneme": 1489,         # 5404 x 5, binary
}


def models():
    return {
        "HypothesisTree": lambda: HypothesisTreeClassifier(),
        "DecisionTree": lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE),
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
            random_state=RANDOM_STATE),
    }


def load(name, data_id):
    """Fetch one suite dataset as (X float64, y int labels), or None with a
    reason when it can't be used as-is (non-numeric or missing values)."""
    bunch = fetch_openml(data_id=data_id, as_frame=False, parser="liac-arff")
    X = np.asarray(bunch.data)
    if X.dtype == object:
        return None, "non-numeric features"
    X = X.astype(np.float64)
    if np.isnan(X).any():
        return None, "missing values"
    y = LabelEncoder().fit_transform(bunch.target)
    return (X, y), None


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


def fmt_time(seconds):
    ms = seconds * 1000
    return f"{ms / 1000:.1f}s" if ms >= 1000 else f"{ms:.0f}ms"


def main():
    warnings.filterwarnings("ignore")
    all_models = models()

    wanted = sys.argv[1:] or list(SUITE)
    unknown = [w for w in wanted if w not in SUITE]
    if unknown:
        sys.exit(f"unknown dataset(s) {unknown}; choose from {sorted(SUITE)}")

    print(f"5-fold stratified CV (random_state={RANDOM_STATE}); "
          f"mean accuracy / mean ROC-AUC (OVR for multiclass) / mean fit time\n")
    print("| Dataset | " + " | ".join(all_models) + " |")
    print("|" + "---|" * (len(all_models) + 1))

    acc_ranks = {m: [] for m in all_models}
    for name in wanted:
        result, reason = load(name, SUITE[name])
        if result is None:
            print(f"| {name} | skipped: {reason} |")
            continue
        X, y = result

        cells = []
        accs = {}
        for model_name, make_model in all_models.items():
            acc, auc, fit_s = cv_scores(make_model, X, y)
            accs[model_name] = acc
            cells.append(f"{acc:.3f} / {auc:.3f} / {fmt_time(fit_s)}")
        shape = f"{X.shape[0]}x{X.shape[1]}"
        print(f"| {name} ({shape}) | " + " | ".join(cells) + " |")

        # Rank models by accuracy on this dataset (1 = best; ties averaged).
        ordered = sorted(accs.values(), reverse=True)
        for model_name, acc in accs.items():
            first = ordered.index(acc) + 1
            last = len(ordered) - ordered[::-1].index(acc)
            acc_ranks[model_name].append((first + last) / 2)

    if any(acc_ranks.values()):
        means = {m: np.mean(r) for m, r in acc_ranks.items() if r}
        print("\nmean accuracy rank (1 = best): "
              + ", ".join(f"{m} {v:.2f}" for m, v in means.items()))


if __name__ == "__main__":
    main()
