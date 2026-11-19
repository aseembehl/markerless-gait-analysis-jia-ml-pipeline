"""Evaluate a saved XGBoost tree model (JSON) with numpy only.

The heel-strike model is trained with XGBoost (outside this repository), but
loading the XGBoost library at inference time is avoided: on macOS it brings
its own OpenMP runtime, which deadlocks PyTorch when both are used in one
process, and it requires a system ``libomp``. A trained gradient-boosted tree
model is fully described by its JSON file, so prediction is reimplemented
here. Supported: ``gbtree`` booster, numerical splits, ``multi:softprob`` /
``multi:softmax`` objectives (as produced by ``XGBClassifier``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class _Tree:
    left: np.ndarray
    right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray  # split condition, or leaf value for leaves
    default_left: np.ndarray
    is_leaf: np.ndarray


def _parse_base_score(raw: str, num_class: int) -> np.ndarray:
    raw = raw.strip()
    if raw.startswith("["):
        vals = np.array(json.loads(raw), dtype=np.float64)
        # XGBoost >= 3 stores per-class intercepts as probabilities; softprob
        # turns them into margins with log().
        return np.log(np.clip(vals, 1e-300, None))
    # Scalar base_score adds the same margin to every class: no effect on softmax.
    return np.full(num_class, float(raw))


class XGBTreeModel:
    """Numpy re-implementation of ``XGBClassifier.predict_proba`` for a JSON model."""

    def __init__(self, path: str | Path):
        d = json.loads(Path(path).read_text())
        learner = d["learner"]
        obj = learner["objective"]["name"]
        if obj not in ("multi:softprob", "multi:softmax"):
            raise ValueError(f"Unsupported XGBoost objective {obj!r}")
        booster = learner["gradient_booster"]
        if booster.get("name") != "gbtree":
            raise ValueError(f"Unsupported XGBoost booster {booster.get('name')!r}")
        params = learner["learner_model_param"]
        self.num_class = int(params["num_class"])
        self.num_feature = int(params["num_feature"])
        self.base_margin = _parse_base_score(params["base_score"], self.num_class)

        model = booster["model"]
        n_trees = len(model["trees"])
        best = learner.get("attributes", {}).get("best_iteration")
        if best is not None and "iteration_indptr" in model:
            n_trees = int(model["iteration_indptr"][int(best) + 1])
        self.tree_class = np.asarray(model["tree_info"][:n_trees], dtype=int)
        self.trees = []
        for t in model["trees"][:n_trees]:
            if any(t.get("split_type", [])):
                raise ValueError("Categorical splits are not supported")
            left = np.asarray(t["left_children"], dtype=np.int64)
            self.trees.append(
                _Tree(
                    left=left,
                    right=np.asarray(t["right_children"], dtype=np.int64),
                    feature=np.asarray(t["split_indices"], dtype=np.int64),
                    threshold=np.asarray(t["split_conditions"], dtype=np.float32),
                    default_left=np.asarray(t["default_left"], dtype=bool),
                    is_leaf=left == -1,
                )
            )

    def _leaf_values(self, tree: _Tree, X: np.ndarray) -> np.ndarray:
        node = np.zeros(len(X), dtype=np.int64)
        rows = np.arange(len(X))
        active = ~tree.is_leaf[node]
        while active.any():
            r, n = rows[active], node[active]
            x = X[r, tree.feature[n]]
            go_left = np.where(np.isnan(x), tree.default_left[n], x < tree.threshold[n])
            node[r] = np.where(go_left, tree.left[n], tree.right[n])
            active = ~tree.is_leaf[node]
        return tree.threshold[node].astype(np.float64)

    def predict_margin(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.num_feature:
            raise ValueError(f"Expected (n, {self.num_feature}) features, got {X.shape}")
        margin = np.tile(self.base_margin, (len(X), 1))
        for tree, cls in zip(self.trees, self.tree_class):
            margin[:, cls] += self._leaf_values(tree, X)
        return margin

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        m = self.predict_margin(X)
        m -= m.max(axis=1, keepdims=True)
        e = np.exp(m)
        return e / e.sum(axis=1, keepdims=True)
