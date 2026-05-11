import numpy as np
import sklearn
from sklearn.utils import resample
import DecisionTree
class RandomForest:
    def __init__(self, n_trees = 100, n_min = 2, max_depth = 100, n_features = None, task = 'classification'):
        self.n_trees = n_trees
        self.n_min = n_min
        self.max_depth = max_depth
        self.n_features = n_features
        self.task = task
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree = DecisionTree.DecisionTree(criterion='cart', n_min=self.n_min, max_depth=self.max_depth, n_features=self.n_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def bootstrap_sample(self, X, y):
        X_sample, y_sample = resample(X, y, replace=True)
        return X_sample, y_sample
    
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        if self.task == 'classification':
            majority_votes = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_preds)
            return majority_votes
        else:
            return np.mean(tree_preds, axis=0)
        