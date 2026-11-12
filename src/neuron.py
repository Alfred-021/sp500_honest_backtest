"""A single artificial neuron (logistic regression) built from scratch in NumPy.
Step 1 of the learning path: neuron -> layer -> network (see README roadmap)."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class Neuron(BaseEstimator, ClassifierMixin):
    def __init__(self, lr=0.1, epochs=300):
        self.lr, self.epochs = lr, epochs

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-9
        Xs = (X - self.mu_) / self.sd_
        n, d = Xs.shape
        self.w_, self.b_ = np.zeros(d), 0.0
        for _ in range(self.epochs):
            p = sigmoid(Xs @ self.w_ + self.b_)        # forward pass
            err = p - y                                # dLoss/dz for cross-entropy
            self.w_ -= self.lr * (Xs.T @ err) / n      # gradient descent step
            self.b_ -= self.lr * err.mean()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = sigmoid(((np.asarray(X, float) - self.mu_) / self.sd_) @ self.w_ + self.b_)
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
