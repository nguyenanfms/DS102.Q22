import numpy as np

class SVM:
    def __init__(self, lr: float = 0.001, epochs: int = 1000, C: float = 1.0):
        self.lr = lr
        self.epochs = epochs
        self.C = C
        self.w = None
        self.b = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, show_progress: bool = False):
        n_samples, n_features = X.shape
        # khởi tạo trọng số w là một mảng 0 có độ dài bằng số lượng đặc trưng và bias b là 0 
        self.w = np.zeros(n_features)
        self.b = 0
        e = range(self.epochs)
        if show_progress:
            from tqdm.auto import tqdm
            e = tqdm(e, total=self.epochs, desc="Training SVM", unit="epoch")

        for e in e:
            for idx, x_i in enumerate(X):
                # tính toán điều kiện của SVM: y_i * (w.x_i - b) >= 1
                # condition tính toán xem điểm dữ liệu có nằm trong margin hay không, true khi điểm dữ liệu nằm ngoài, và nằm trên margin và false khi điểm dữ liệu nằm trong margin hoặc trên đường biên
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    # nếu điều kiện thỏa mãn cập nhật trọng số w và bias b theo công thức w = w - lr * (2 * lambda * w) và b = b - lr * 0 (gradient descent)
                    self.w -= self.lr * self.w
                    self.b -= self.lr * 0
                else:
                    # nếu điều kiện không thỏa mãn
                    self.w -= self.lr * (self.w - self.C*y[idx]*x_i)
                    # Với y*(w.x + b) < 1, gradient theo b là -C*y => cập nhật GD: b = b + lr*C*y
                    self.b += self.lr * self.C*y[idx]
                    
    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = X @ self.w +self.b
        return np.sign(y_pred)  
    
    def evaluate_metrics(self, X: np.ndarray, y: np.ndarray):
        y_pred = self.predict(X)

        tp = np.sum((y_pred == 1) & (y == 1))
        fp = np.sum((y_pred == 1) & (y == -1))
        tn = np.sum((y_pred == -1) & (y == -1))
        fn = np.sum((y_pred == -1) & (y == 1))

        eps = 1e-12
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1_score = 2 * (precision * recall) / (precision + recall + eps)
        acuracy = (tp + tn) / len(y)

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': acuracy
        }