# gmm_model.py
import numpy as np


def multivariate_normal_pdf(X, mean, cov):
    D = X.shape[1]
    cov = cov + 1e-6 * np.eye(D)  # Tránh ma trận hiệp biến bị suy biến
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    diff = X - mean
    exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)
    return (1.0 / np.sqrt(((2 * np.pi) ** D) * det)) * np.exp(exponent)


class GaussianMixtureModel:

    def __init__(self, K, max_iters=100, tol=1e-4):
        self.K = K
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):
        N, D = X.shape
        self.weights = np.ones(self.K) / self.K
        self.means = X[np.random.choice(N, self.K, replace=False)]
        self.covariances = np.array([np.eye(D) for _ in range(self.K)])

        old_log_likelihood = 0

        for i in range(self.max_iters):
            # E-step
            responsibilities = np.zeros((N, self.K))
            for k in range(self.K):
                responsibilities[:, k] = self.weights[
                    k] * multivariate_normal_pdf(
                        X, self.means[k], self.covariances[k])

            sum_resp = np.sum(responsibilities, axis=1, keepdims=True)
            sum_resp = np.where(sum_resp == 0, 1e-15, sum_resp)
            responsibilities /= sum_resp

            # M-step
            N_k = np.sum(responsibilities, axis=0)

            for k in range(self.K):
                if N_k[k] == 0:
                    continue
                self.means[k] = (
                    np.sum(responsibilities[:, k:k + 1] * X, axis=0) / N_k[k])
                diff = X - self.means[k]
                self.covariances[k] = (
                    diff.T @ (responsibilities[:, k:k + 1] * diff) / N_k[k])

            self.weights = N_k / N

            log_likelihood = np.sum(np.log(sum_resp))
            if np.abs(log_likelihood - old_log_likelihood) < self.tol:
                print(f"GMM hội tụ tại vòng lặp thứ: {i}")
                break
            old_log_likelihood = log_likelihood

    def predict(self, X):
        N = X.shape[0]
        responsibilities = np.zeros((N, self.K))
        for k in range(self.K):
            responsibilities[:, k] = self.weights[
                k] * multivariate_normal_pdf(X, self.means[k],
                                             self.covariances[k])
        return np.argmax(responsibilities, axis=1)