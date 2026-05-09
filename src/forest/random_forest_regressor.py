import numpy as np
import pandas as pd
from src.tree.decision_tree_regressor import DecisionTreeRegressor
from src.forest.bootstrap import Bootstrap

class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, max_features="sqrt", random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
    
    def fit(self, X, y):
        self.trees = []
        X = X.copy()
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]
        y = y.copy() if isinstance(y, pd.Series) else pd.Series(y, index=X.index)
        
        for i in range(self.n_estimators):
            bootstrap = Bootstrap(X, random_state=self.random_state + i)
            X_sample, sample_positions = bootstrap.create_bootstrap_sample()
            y_sample = y.iloc[sample_positions].reset_index(drop=True)
            X_sample = X_sample.reset_index(drop=True)

            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, max_features=self.max_features, random_state=self.random_state + i)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

        self.feature_importances_ = self._calculate_feature_importances()

    def _calculate_feature_importances(self):
        if len(self.trees) == 0:
            return np.array([])

        importances = np.array([tree.feature_importances_ for tree in self.trees])
        mean_importances = np.mean(importances, axis=0)
        total_importance = np.sum(mean_importances)

        if total_importance > 0:
            mean_importances = mean_importances / total_importance

        return mean_importances

    
    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_predictions, axis=0)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
        return r2_score
