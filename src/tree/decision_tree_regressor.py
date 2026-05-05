import numpy as np
import pandas as pd
from .decision_tree_node import TreeNode

class DecisionTreeRegressor:
    def __init__(self, max_depth=10, min_samples_split=5, min_samples_leaf=2, max_features="sqrt", random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self.feature_importances_ = None

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        y = np.array(y)

        self.n_features_ = X.shape[1]

        self.feature_importances_ = np.zeros(self.n_features_)

        self.root = self._build_tree(X, y, 0)

        total = np.sum(self.feature_importances_)

        if total > 0:
            self.feature_importances_ /= total

    def _build_tree(self, X, y, depth):
        if (self.max_depth is not None and depth >= self.max_depth or len(y) < self.min_samples_split) or len(y) < self.min_samples_leaf:
            return TreeNode(value=np.mean(y), is_leaf=True)
        
        value = np.mean(y)
        if self._should_stop(y, depth):
            return TreeNode(value=value, is_leaf=True)

        feature_indices = self._sample_feature_indices()
        best_feature, best_threshold, best_gain = self._best_split(X, y, feature_indices)

        if best_feature is None or best_gain <= 0:
            return TreeNode(value=value, is_leaf=True)

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        if len(y[left_mask]) < self.min_samples_leaf or len(y[right_mask]) < self.min_samples_leaf:
            return TreeNode(value=value, is_leaf=True)

        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        self.feature_importances_[best_feature] += best_gain

        return TreeNode(
            feature_index=best_feature,
            feature_name=self.feature_names[best_feature],
            threshold=best_threshold,
            left=left,
            right=right,
            value=value,
            is_leaf=False,
        )

    def _should_stop(self, y, depth):
        reached_max_depth = self.max_depth is not None and depth >= self.max_depth
        too_few_samples = len(y) < self.min_samples_split
        pure_node = np.var(y) == 0
        return reached_max_depth or too_few_samples or pure_node

    def _sample_feature_indices(self):
        if self.max_features == "sqrt":
            n_selected = max(1, int(np.sqrt(self.n_features_)))
        elif self.max_features == "third":
            n_selected = max(1, self.n_features_ // 3)
        elif self.max_features == "all" or self.max_features is None:
            n_selected = self.n_features_
        elif isinstance(self.max_features, int):
            n_selected = min(self.n_features_, max(1, self.max_features))
        else:
            raise ValueError("max_features must be 'sqrt', 'third', 'all', None, or an integer")

        self.rng = np.random.default_rng(self.random_state)
        return self.rng.choice(self.n_features_, size=n_selected, replace=False)

    def _best_split(self, X, y, feature_indices):
        best_mse = float("inf")
        best_feature = None
        best_threshold = None
        best_gain = -1

        for feature_index in feature_indices:
            values = np.unique(X[:, feature_index])
            if len(values) == 1:
                continue
            thresholds = (values[:-1] + values[1:]) / 2
            for threshold in thresholds:
                left_indices = X[:, feature_index] < threshold
                right_indices = X[:, feature_index] >= threshold
                if len(y[left_indices]) < self.min_samples_leaf or len(y[right_indices]) < self.min_samples_leaf:
                    continue
                mse = self._calculate_mse(y[left_indices], y[right_indices])
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature_index
                    best_threshold = threshold
                    best_gain = np.var(y) - mse

        return best_feature, best_threshold, best_gain

    def _mse(self, y):
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)

    def _calculate_mse(self, left_y, right_y):
        if len(left_y) == 0 or len(right_y) == 0:
            return float("inf")
        left_mean = np.mean(left_y)
        right_mean = np.mean(right_y)
        mse_left = np.mean((left_y - left_mean) ** 2)
        mse_right = np.mean((right_y - right_mean) ** 2)
        return (len(left_y) * mse_left + len(right_y) * mse_right) / (len(left_y) + len(right_y))

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_single(x, self.root) for x in X])

    def _predict_single(self, x, node):
        if node.is_leaf:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

    def bootstrap_sample(self, X, y):
        n_samples = len(y)
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    
        
