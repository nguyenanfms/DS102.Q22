import numpy as np
from tqdm import tqdm
import sklearn
from sklearn.metrics import precision_score, recall_score, f1_score

class SoftmaxRegression:
    def __init__(self, epoch: int, lr :float):
        self.epoch = epoch
        self.lr = lr

        self.w = None
        self.losses = []

    def softmax(self, z: np.ndarray):
        z_shift = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shift)
        denum = np.sum(exp_z, axis=1, keepdims=True)
        return exp_z / denum

    def loss(self, Y:np.ndarray, Y_hat:np.ndarray) -> float:
        eps = 1e-15
        return -np.mean(np.sum(Y * np.log(Y_hat + eps), axis=1))

    def fit(self, X:np.ndarray, Y:np.ndarray):
        N, d = X.shape
        if Y.ndim == 1:
            K = int(np.max(Y)) + 1
            Y = np.eye(K)[Y]
        else:
            K = Y.shape[1]

        self.w = np.zeros((d, K), dtype=np.float64)
        for e in tqdm(range(self.epoch), desc="Training"):
            #forward pass
            Y_hat = self.predict(X)
            delta_y = (Y_hat - Y)

            #backward pass
            gradient = (X.T @ delta_y) / N
            self.w -= self.lr * gradient

            L = self.loss(Y, Y_hat)
            self.losses.append(L)
    
    def evaluate(self, Y, Y_hat) -> dict:
        precision =precision_score(Y, Y_hat, average='macro')
        recall = recall_score(Y, Y_hat, average='macro')
        f1 = f1_score(Y, Y_hat, average='macro')
        return {"precision": precision, "recall": recall, "f1_score": f1}

    def predict(self, X:np.ndarray) -> np.ndarray:
        Y_hat = X @ self.w
        return self.softmax(Y_hat)