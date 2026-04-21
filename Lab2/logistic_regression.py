import numpy as np
from tqdm import tqdm
import sklearn
from sklearn.metrics import precision_score, recall_score, f1_score

class LogisticRegression:
    def __init__(self, epoch: int, lr :float):
        self.epoch = epoch
        self.lr = lr

        self.w = None
        self.losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def loss(self, Y:np.ndarray, Y_hat:np.ndarray) -> float:
        L = (1 - Y)*np.log(1 - Y_hat + 10e-15) + Y*np.log(Y_hat + 10e-15)
        return -L.mean()

    def fit(self, X:np.ndarray, Y:np.ndarray):
        N, d = X.shape
        self.w = np.zeros((d, 1), dtype=np.float64)
        e = 0
        for e in tqdm(range(self.epoch), desc="Training"):
            #forward pass
            Y_hat = self.predict(X)
            delta_y = (Y_hat - Y) 

            #backward pass
            gradient = delta_y.T @ X 
            self.w -= self.lr * gradient.T 

            L = self.loss(Y, Y_hat)
            self.losses.append(L)
    
    def evaluate(self, Y, Y_hat) -> dict:
        precision =precision_score(Y, Y_hat, average='binary')
        recall = recall_score(Y, Y_hat, average='binary')
        f1 = f1_score(Y, Y_hat, average='binary')
        return {"precision": precision, "recall": recall, "f1_score": f1}

    def predict(self, X:np.ndarray) -> np.ndarray:
        Y_hat = X @ self.w #(N,1)
        return self.sigmoid(Y_hat)