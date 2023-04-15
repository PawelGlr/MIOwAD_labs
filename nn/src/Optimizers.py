from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Collection

import numpy as np

from nn.src.Layers import Layer, TrainableLayer


class Optimizer(ABC):
    """
    Base class for optimizers.
    """

    def __init__(self, layers: Collection[Layer], lr: float = 0.01):
        self.lr = lr
        self.layers = [layer for layer in layers if isinstance(layer, TrainableLayer)]

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    """
    Stochastic gradient descent optimizer.
    |w = w - lr * dw|
    """

    def __init__(self, layers: Collection[Layer], lr: float = 0.01):
        super().__init__(layers, lr)

    def step(self):
        for layer in self.layers:
            layer_gradients = layer.get_gradients()
            for key in layer_gradients:
                layer_gradients[key] *= -self.lr
            layer.change_parameters(layer_gradients)


class MomentumOptimizer(Optimizer):
    """
    Momentum optimizer.
    |v = momentum * v - lr * dw|
    |w = w + v|
    """

    def __init__(self, layers: Collection[Layer], lr: float = 0.01, momentum: float = 0.9):
        super().__init__(layers, lr)

        self.momentum = momentum
        self.velocity = []
        for layer in self.layers:
            layer_velocity = {}
            layer_gradients = layer.get_gradients()
            for key in layer_gradients:
                layer_velocity[key] = np.zeros_like(layer_gradients[key])
            self.velocity.append(layer_velocity)

    def step(self):
        for layer, layer_velocity in zip(self.layers, self.velocity):
            layer_gradients = layer.get_gradients()
            for key in layer_gradients:
                layer_velocity[key] = self.momentum * layer_velocity[key] - self.lr * layer_gradients[key]
            layer.change_parameters(layer_velocity)


class Adam(Optimizer):
    """
    Adaptive momentum optimizer.
    |m = beta_1 * m + (1 - beta_1) * dw|
    |v = beta_2 * v + (1 - beta_2) * dw^2|
    |m_hat = m / (1 - beta_1^t)|
    |v_hat = v / (1 - beta_2^t)|
    |w = w - lr * m_hat / (sqrt(v_hat) + eps)|
    """

    def __init__(self, layers: Collection[Layer], lr: float = 0.01, beta_1: float = 0.9, beta_2: float = 0.999,
                 eps: float = 1e-8):
        super().__init__(layers, lr)

        self.eps = eps
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.first_moment = []
        self.second_moment = []

        for layer in self.layers:
            layer_first_moment = {}
            layer_second_moment = {}
            layer_gradients = layer.get_gradients()
            for key in layer_gradients:
                layer_first_moment[key] = np.zeros_like(layer_gradients[key])
                layer_second_moment[key] = np.zeros_like(layer_gradients[key])
            self.first_moment.append(layer_first_moment)
            self.second_moment.append(layer_second_moment)

    def step(self):
        for layer, layer_first_moment, layer_second_moment in zip(self.layers, self.first_moment, self.second_moment):
            layer_gradients = layer.get_gradients()
            layer_first_moment_adj = {}
            layer_second_moment_adj = {}
            change = {}
            for key in layer_gradients:
                layer_first_moment[key] = self.beta_1 * layer_first_moment[key] - (1 - self.beta_1) * layer_gradients[
                    key]
                layer_second_moment[key] = self.beta_2 * layer_second_moment[key] + (1 - self.beta_2) * layer_gradients[
                    key] ** 2
                layer_first_moment_adj[key] = layer_first_moment[key] / (1 - self.beta_1)
                layer_second_moment_adj[key] = layer_second_moment[key] / (1 - self.beta_2)
                change[key] = self.lr * layer_first_moment_adj[key] / np.sqrt(layer_second_moment_adj[key] + self.eps)
            layer.change_parameters(change)
