from __future__ import annotations

import numpy as np


class MSE:
    """
    Mean squared error loss function.
    |loss = (y - y_hat)^2|
    """

    def __call__(self, y: np.ndarray, y_hat: np.ndarray) -> int:
        self.last_y = y
        self.last_y_hat = y_hat
        return ((y.flatten() - y_hat.flatten()) ** 2).mean()

    def derriv(self) -> np.ndarray:
        return 2 * (self.last_y_hat.flatten() - self.last_y.flatten()).reshape(-1, 1) / self.last_y.shape[0]


class CrossEntropyLoss:
    """
    Cross entropy loss function.
    |loss = -sum(y * log(y_hat))|
    """

    def __call__(self, y: np.ndarray, y_hat: np.ndarray) -> int:
        self.last_y = y
        self.last_y_hat = y_hat
        return -(y * np.log(y_hat)).sum(axis=1).mean()

    def derriv(self) -> np.ndarray:
        """
        The derriv of the cross entropy loss function is not implemented for the general case. Only to be used with Softmax on last layer.
        """
        return (self.last_y_hat.flatten() - self.last_y.flatten()).reshape(-1, self.last_y.shape[1]) / \
               self.last_y.shape[0]


class BinaryCrossEntropy:
    """
    Binary cross entropy loss function.
    |loss = -sum(y * log(y_hat) + (1 - y) * log(1 - y_hat))|
    """

    def __call__(self, y: np.ndarray, y_hat: np.ndarray) -> int:
        self.eps = 1e-8
        self.last_y = y
        self.last_y_hat = y_hat
        return -(y * np.log(y_hat + self.eps).flatten() + (1 - y) * np.log(1 - y_hat + self.eps).flatten()).mean()

    def derriv(self) -> np.ndarray:
        return (-(self.last_y.flatten() / (self.last_y_hat.flatten() + self.eps)) + (1 - self.last_y.flatten()) / (
                1 - self.last_y_hat.flatten() + self.eps)).reshape(-1, 1) / self.last_y.shape[0]
