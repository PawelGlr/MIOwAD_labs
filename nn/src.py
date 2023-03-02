from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import uniform
import math
from typing import Tuple, Collection


class LinearLayer:
    def __init__(self, in_shape: int, out_shape: int, weights: np.ndarray = None, dist=None, bias: bool = True,
                 bias_values: np.ndarray = None):
        if dist is None:
            k = math.sqrt(1 / in_shape)
            dist = uniform(loc=-k, scale=2 * k)
        if weights is None:
            weights = dist.rvs((in_shape, out_shape))
        if bias and bias_values is None:
            bias_values = dist.rvs(out_shape)
        self.weights = weights
        if bias:
            self.bias_weights = bias_values
        self.bias = bias

    def __call__(self, X: np.ndarray):
        if self.bias:
            return np.dot(X, self.weights) + self.bias_weights
        else:
            return np.dot(X, self.weights)


class MLP:
    def __init__(self, layers: Collection):
        self.layers = list(layers)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape((-1, 1))
        for layer in self.layers:
            X = layer(X)
        return X


def sigmoid(X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-X))


def relu(X: np.ndarray) -> np.ndarray:
    return np.where(X < 0, 0, X)


def mse(y: np.ndarray, y_hat: np.ndarray) -> int:
    return ((y.flatten() - y_hat.flatten()) ** 2).mean()


def make_simple_sequential(shapes: Collection, activation: callable = sigmoid):
    layers = []
    for (in_shape, out_shape) in zip(shapes[:-1], shapes[1:]):
        layers.append(LinearLayer(in_shape=in_shape, out_shape=out_shape))
        layers.append(activation)
    layers = layers[:-1]
    return MLP(layers)
# %%
