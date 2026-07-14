import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split

from hypothesis_tree import HypothesisTreeClassifier


@pytest.fixture(scope="module")
def iris_split():
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


def test_fit_predict_iris(iris_split):
    X_train, X_test, y_train, y_test = iris_split
    clf = HypothesisTreeClassifier().fit(X_train, y_train)

    train_acc = np.mean(clf.predict(X_train) == y_train)
    test_acc = np.mean(clf.predict(X_test) == y_test)
    assert train_acc >= 0.95  # error-driven carving should nail the train set
    assert test_acc >= 0.80


def test_predict_proba_is_a_distribution(iris_split):
    X_train, X_test, y_train, _ = iris_split
    clf = HypothesisTreeClassifier().fit(X_train, y_train)

    probas = clf.predict_proba(X_test)
    assert probas.shape == (len(X_test), len(clf.classes_))
    assert np.all(probas >= 0) and np.all(probas <= 1)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-9)

    # predict and predict_proba must agree on ties-free rows
    pred_from_proba = clf.classes_[np.argmax(probas, axis=1)]
    pred = clf.predict(X_test)
    agree = np.mean(pred_from_proba == pred)
    assert agree >= 0.95


def test_deterministic(iris_split):
    X_train, X_test, y_train, _ = iris_split
    a = HypothesisTreeClassifier().fit(X_train, y_train)
    b = HypothesisTreeClassifier().fit(X_train, y_train)
    assert len(a.tree_) == len(b.tree_)
    np.testing.assert_array_equal(a.predict(X_test), b.predict(X_test))


def test_sklearn_clone_and_params():
    clf = HypothesisTreeClassifier(max_epochs=5, softness=0.2, verbose=0)
    cloned = clone(clf)
    assert cloned.get_params()["max_epochs"] == 5
    assert cloned.get_params()["softness"] == 0.2

    clf.set_params(max_epochs=7)
    assert clf.max_epochs == 7


def test_string_labels():
    X, y = load_iris(return_X_y=True)
    names = np.array(["setosa", "versicolor", "virginica"])
    y_str = names[y]
    clf = HypothesisTreeClassifier().fit(X, y_str)
    pred = clf.predict(X[:10])
    assert set(pred) <= set(names)
    assert list(clf.classes_) == sorted(names)


def test_single_class():
    X = np.random.RandomState(0).rand(20, 3)
    y = np.zeros(20, dtype=int)
    clf = HypothesisTreeClassifier().fit(X, y)
    assert np.all(clf.predict(X) == 0)
    probas = clf.predict_proba(X)
    np.testing.assert_allclose(probas, 1.0)


def test_wine_reasonable_accuracy():
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)
    clf = HypothesisTreeClassifier().fit(X_train, y_train)
    assert np.mean(clf.predict(X_test) == y_test) >= 0.70


def test_tree_is_inspectable(iris_split):
    X_train, _, y_train, _ = iris_split
    clf = HypothesisTreeClassifier().fit(X_train, y_train)
    tree = clf.tree_
    assert len(tree) >= 3  # at least the root plus carved children
    # Every non-root cluster is wired to a parent that exists.
    for child, parent in tree.parents.items():
        assert parent == -1 or 0 <= parent < len(tree)
    # Anchors are actual training rows (float32-cast comparison).
    anchors = np.asarray(tree.X[1], dtype=np.float32)
    train32 = np.asarray(X_train, dtype=np.float32)
    assert np.any(np.all(train32 == anchors, axis=1))


def test_unfitted_raises():
    from sklearn.exceptions import NotFittedError
    with pytest.raises(NotFittedError):
        HypothesisTreeClassifier().predict(np.zeros((2, 3)))


def test_closer_strategy(iris_split):
    X_train, X_test, y_train, y_test = iris_split
    clf = HypothesisTreeClassifier(match_strategy="closer").fit(X_train, y_train)
    assert np.mean(clf.predict(X_test) == y_test) >= 0.70

    # Deterministic, like the default strategy.
    again = HypothesisTreeClassifier(match_strategy="closer").fit(X_train, y_train)
    np.testing.assert_array_equal(clf.predict(X_test), again.predict(X_test))


def test_unknown_strategy_raises(iris_split):
    X_train, _, y_train, _ = iris_split
    with pytest.raises(ValueError, match="match_strategy"):
        HypothesisTreeClassifier(match_strategy="nope").fit(X_train, y_train)
