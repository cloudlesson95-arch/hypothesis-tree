"""Minimal end-to-end example: fit, predict, and inspect the tree."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from hypothesis_tree import HypothesisTreeClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

clf = HypothesisTreeClassifier(verbose=1)
clf.fit(X_train, y_train)

acc = np.mean(clf.predict(X_test) == y_test)
print(f"\ntest accuracy: {acc:.3f}")
print(f"clusters: {len(clf.tree_)} over {clf.n_epochs_} epochs")

# Every prediction is traceable: route one sample by hand.
tree = clf.tree_
sample = X_test[0]
pred, conf = tree.forward(sample)
path = tree.history[0]["path"]
print(f"\nsample {sample} -> predicted class {pred[0]} (confidence {conf[0]:.2f})")
print(f"descent path (cluster indices): {path}")
leaf = path[-1]
print(f"matched exemplar (cluster {leaf}): {np.asarray(tree.X[leaf])}")
print(f"active features: {np.where(tree.masks[leaf])[0].tolist()}")
