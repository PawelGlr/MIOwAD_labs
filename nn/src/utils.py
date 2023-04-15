from __future__ import annotations

from typing import Collection, Any

import numpy as np

from nn.src.Layers import Sigmoid, LinearLayer
from nn.src.Networks import MLP


class DataLoader:
    """
    Data loader class.
    Last batch will be smaller if batch_size does not divide the number of samples.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int, seed=None):
        """
        Initialize the data loader.
        :param X: Datapoints to use
        :param y: Labels to use
        :param batch_size: batch size to use (use number of samples, or bigger if you want to use all samples)
        :param seed: seed to use for random sampling
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.current_idx = 0
        self.length = self.X.shape[0]

    def __getitem__(self, item):
        return (self.X[item, :], self.y[item, :])

    def __len__(self):
        return self.length

    def __iter__(self):
        """
        Iterate over the data loader.
        :return: touple of batch of X and batch of y
        """
        self.current_idx = 0
        order = self.rng.choice(self.length, self.length, replace=False)
        self.X = self.X[order]
        self.y = self.y[order]
        while self.current_idx < self.length:
            self.current_idx += self.batch_size
            yield (self.X[self.current_idx - self.batch_size:self.current_idx],
                   self.y[self.current_idx - self.batch_size:self.current_idx])


def make_simple_sequential(shapes: Collection[int], activation: Any = Sigmoid) -> MLP:
    """
    Creates a simple sequential model.
    :param shapes: Collection of shapes of layers. The first shape is the input shape, the last is the output shape. Numbers between are numbers of neurons in hidden layers.
    :param activation: Activation function for hidden layers. Used for all hidden layers.
    :return: Sequential model.
    """
    layers = []
    for (in_shape, out_shape) in zip(shapes[:-1], shapes[1:]):
        layers.append(LinearLayer(in_shape=in_shape, out_shape=out_shape))
        layers.append(activation())
    layers = layers[:-1]
    return MLP(layers)


class ScalingTransformer:
    def __init__(self, a: float = -1, b: float = 1):
        self.translation = None
        self.multiplier = None
        self.a = a
        self.b = b

    def fit(self, array: np.ndarray):
        self.multiplier = (self.b - self.a) / (array.max() - array.min())
        self.translation = self.a - array.min() * self.multiplier
        return self

    def transform(self, array: np.ndarray):
        return array * self.multiplier + self.translation

    def fit_transform(self, array: np.ndarray):
        return self.fit(array).transform(array)

    def inverse_transform(self, array: np.ndarray):
        return (array - self.translation) / self.multiplier
