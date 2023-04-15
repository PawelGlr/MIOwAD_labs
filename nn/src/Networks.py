from __future__ import annotations

from typing import Collection

import numpy as np

from nn.src.Layers import Layer


class MLP:
    """
    Multi-layer perceptron.
    calls the layers in the order they are given in the constructor.
    """

    def __init__(self, layers: Collection[Layer]):
        """
        Initialize the MLP.
        :param layers: layers of the MLP
        """
        self.train = True
        self.layers = list(layers)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass of the MLP.
        Calls the layers in the order
        :param X: input values (batch size x input shape) or (input shape)
        :return: output values (batch size x output shape)
        """
        if X.ndim == 1:
            X = X.reshape((-1, 1))
        for layer in self.layers:
            X = layer(X)
        return X

    def set_train(self, train: bool = True):
        """
        Set the train mode of the MLP.
        Calls the set_train method of all layers.
        :param train: True if the MLP is set to train mode, False if the MLP is set to test mode
        """
        self.train = train
        for layer in self.layers:
            try:
                layer.set_train(train)
            except AttributeError:
                pass

    def backward(self, gradient: np.ndarray):
        """
        Backward pass of the MLP.
        Calls the backward method of all layers in reverse order.
        Gradients are stored in the layers. Use the get_gradients method to get the gradients.
        :param gradient: gradient of the loss function (batch size x output shape)
        :return: gradient of the input (batch size x input shape) can be ignored in most cases
        """
        for layer in reversed(self.layers):
            # print(gradient.shape, layer)
            gradient = layer.derriv(gradient)
        return gradient
