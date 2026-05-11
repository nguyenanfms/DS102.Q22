import numpy as np
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, criterion='cart', n_min = 2, max_depth=100,n_features = None):
        self.criterion = criterion
        self.n_min = n_min
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        # Điều kiện dừng:
        # - Quá độ sâu tối đa
        # - Số mẫu nhỏ hơn n_min (ko đủ mẫu để chia tiếp)
        # - Tất cả mẫu cùng nhãn
        if (depth >= self.max_depth) or (n_samples < self.n_min) or (n_labels == 1):
            leaf_value = self.calculate_leaf_value(y)
            return Node(value=leaf_value)

        # Tìm cách chia tốt nhất
        best_feature, best_threshold = self.best_split(X, y, n_features)

        X_column = X[:, best_feature]
        left_idx, right_idx = self.split(X_column, best_threshold)
        
        # Đệ quy cho cành trái và cành phải
        left = self.build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right = self.build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feature, best_threshold, left, right)

    def best_split(self, X, y, total_features):
        best_score = float('inf') if self.criterion == 'cart' else -1
        split_idx, splits_threshold = None, None

        if self.n_features is None:
            num_features = total_features
        else:
            num_features = min(self.n_features, total_features)
        feature_indices = np.random.choice(total_features, num_features, replace=False)

        for feature_idx in feature_indices:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
    
            for thr in thresholds:
                left_idx, right_idx = self.split(X_column, thr)

                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue
                
                left, right = y[left_idx], y[right_idx]
                if self.criterion == 'cart':
                    # Tìm cart loss nhỏ nhất 
                    loss = self.cart_loss(left, right)
                    if loss < best_score:
                        best_score = loss
                        split_idx = feature_idx
                        splits_threshold = thr
                else:
                    # Tìm IG lớn nhất 
                    gain = self.if_gain(y, left, right)
                    if gain > best_score:
                        best_score = gain
                        split_idx = feature_idx
                        splits_threshold = thr
                        
        return split_idx, splits_threshold
    def if_gain(self, parent, left, right):
        # Tính H(S)
        parent_entropy = self.entropy(parent)
        # Tính trung bình cộng số entropy của node con H(x,S)
        n = len(parent)
        n_left, n_right = len(left), len(right)
        e_left, e_right = self.entropy(left), self.entropy(right)
        h = (n_left / n) * e_left + (n_right / n) * e_right
        #Tính G(x,S) = H(S) - H(x,S)
        return parent_entropy - h
    def entropy(self, y):
        counts = np.bincount(y)
        ps = counts / len(y)
        return -np.sum(ps * np.log2(ps + 1e-9))  # Thêm epsilon để tránh log(0)
    def cart_loss(self, left, right):
        # Tính Gini(left) và Gini(right)
        gini_left = self.gini(left)
        gini_right = self.gini(right)
        # Tính trung bình cộng số gini của node con G(x,S)
        n = len(left) + len(right)
        return (len(left) / n) * gini_left + (len(right) / n) * gini_right
    def gini(self, y):
        counts = np.bincount(y)
        ps = counts / len(y)
        return 1 - np.sum(ps ** 2)
    def split(self, X_column, threshold):
        left_idx = np.where(X_column <= threshold)[0]
        right_idx = np.where(X_column > threshold)[0]
        return left_idx, right_idx
    def calculate_leaf_value(self, y):
        most_common = np.bincount(y).argmax()
        return most_common
    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root) for x in X])
    def traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        feature_value = x[node.feature_index]
        if feature_value <= node.threshold:
            return self.traverse_tree(x, node.left)
        else:
            return self.traverse_tree(x, node.right)
