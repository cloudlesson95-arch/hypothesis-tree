"""hypothesis-tree: an exemplar-based tree classifier.

Public API: `HypothesisTreeClassifier` (scikit-learn compatible). The fitted
`tree_` attribute exposes the cluster tree for inspection.
"""
from .classifier import HypothesisTreeClassifier
from .tree import HypothesisTree

__version__ = "0.1.0"
__all__ = ["HypothesisTreeClassifier", "HypothesisTree", "__version__"]
