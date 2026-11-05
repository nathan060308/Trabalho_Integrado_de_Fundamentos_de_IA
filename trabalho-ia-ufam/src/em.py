# src/em.py
import numpy as np


class GaussianEM:
    """Algoritmo Expectation-Maximization (EM) para misturas de Gaussianas (GMM)

    com variável oculta Z (K componentes), implementado a partir do zero.
    """

    def __init__(self, n_components=2, max_iter=50, tol=1e-4, seed=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.weights = None
        self.means = None
        self.covariances = None
        self.log_likelihood_history = []

    def fit(self, X):
        np.random.seed(self.seed)
        n_samples, n_features = X.shape
        K = self.n_components

        # 1. Inicialização dos parâmetros (pesos, médias e covariâncias)
        self.weights = np.ones(K) / K
        random_idx = np.random.choice(n_samples, K, replace=False)
        self.means = X[random_idx].copy()
        self.covariances = [
            np.eye(n_features) * np.var(X, axis=0) for _ in range(K)
        ]

        responsibilities = np.zeros((n_samples, K))

        for it in range(self.max_iter):
            # Expectation Step (E-step)
            for k in range(K):
                diff = X - self.means[k]
                cov = self.covariances[k] + 1e-6 * np.eye(
                    n_features
                )  # Regularização numérica
                inv_cov = np.linalg.inv(cov)
                det_cov = np.linalg.det(cov)

                norm_const = 1.0 / (
                    (2 * np.pi) ** (n_features / 2) * np.sqrt(det_cov)
                )
                exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
                responsibilities[:, k] = (
                    self.weights[k] * norm_const * np.exp(exponent)
                )

            # Normalização das responsabilidades r[k, x]
            sum_resp = (
                np.sum(responsibilities, axis=1, keepdims=True) + 1e-12
            )
            responsibilities /= sum_resp

            # Cálculo da Log-Verossimilhança
            log_lh = np.sum(np.log(sum_resp.flatten()))
            self.log_likelihood_history.append(log_lh)

            # Maximization Step (M-step)
            N_k = np.sum(responsibilities, axis=0)
            for k in range(K):
                self.weights[k] = N_k[k] / n_samples
                self.means[k] = (
                    np.sum(responsibilities[:, k : k + 1] * X, axis=0) / N_k[k]
                )
                diff = X - self.means[k]
                self.covariances[k] = (
                    (responsibilities[:, k : k + 1] * diff).T @ diff
                ) / N_k[k]

            # Teste de convergência
            if (
                it > 0
                and abs(
                    self.log_likelihood_history[-1]
                    - self.log_likelihood_history[-2]
                )
                < self.tol
            ):
                break

        return responsibilities

    def predict_clusters(self, X):
        resp = self.fit(X)
        return np.argmax(resp, axis=1)