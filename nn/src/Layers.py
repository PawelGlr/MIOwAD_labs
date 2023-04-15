from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import uniform


class Layer(ABC):
    """
    Base class for all layers in the neural network.
    Should implement __call__ and derriv methods.
    """

    def __init__(self):
        self.train = True

    @abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass of the layer.

        :param X: Input values.
        :return: Output values.
        """
        pass

    @abstractmethod
    def derriv(self, next_gradient: np.ndarray) -> np.ndarray:
        pass

    def set_train(self, train: bool = True):
        """Set the layer to train or test mode.

        :param train: True if the layer is in train mode, False if in test mode.
        """
        self.train = train


class TrainableLayer(Layer):
    """
    Base class for all layers that have parameters that can be trained.
    """

    @abstractmethod
    def get_gradients(self) -> dict[str, np.ndarray]:
        """
        Get the gradients of the parameters of the layer.
        :return: dict[str, np.ndarray]
        """

        pass

    @abstractmethod
    def get_parameters(self) -> dict[str, np.ndarray]:
        """
        Get the parameters of the layer.
        :return: dict[str, np.ndarray]
        """
        pass

    @abstractmethod
    def change_parameters(self, changes: dict[str, np.ndarray]):
        """
        Change the parameters of the layer.
        Add the changes to the current parameters.
        :param changes: dict[str, np.ndarray]
        """
        pass


class LinearLayer(TrainableLayer):
    """
    Linear layer.
    |y = X * W + b|

    """

    def __init__(self, in_shape: int, out_shape: int, weights: np.ndarray = None, dist="uniform", var="Glorot",
                 bias: np.ndarray = None, l2: float = 0.0, l1: float = 0.0):
        """

        :param in_shape: input shape (batch size not included)
        :param out_shape: output shape (batch size not included)
        :param weights: initial weights if None weights are initialized according to the dist and var parameters
        :param dist: distribution to use for initialization of weights "uniform" or "normal"  used only if weights is None
        :param var: variance to use for initialization of weights "Glorot", "Xavier", "He", "Lecun"  used only if weights is None
        :param bias: initial bias if None bias is initialized to 0
        :param l2: l2 regularization parameter
        :param l1: l1 regularization parameter
        """
        super().__init__()
        self.bias_gradient = None
        self.gradient = None
        self.l2 = l2
        self.l1 = l1

        if weights is None:
            if dist == "uniform":
                if var == "Glorot" or var == "Xavier":
                    self.weights = uniform.rvs(-math.sqrt(6 / (in_shape + out_shape)),
                                               math.sqrt(6 / (in_shape + out_shape)),
                                               size=(in_shape, out_shape))
                elif var == "He":
                    self.weights = uniform.rvs(-math.sqrt(6 / in_shape), math.sqrt(6 / in_shape),
                                               size=(in_shape, out_shape))
                elif var == "Lecun":
                    self.weights = uniform.rvs(-math.sqrt(3 / in_shape), math.sqrt(3 / in_shape),
                                               size=(in_shape, out_shape))
                else:
                    raise ValueError("Invalid distribution")
            elif dist == "normal":
                if var == "Glorot" or var == "Xavier":
                    self.weights = np.random.normal(0, math.sqrt(2 / (in_shape + out_shape)),
                                                    size=(in_shape, out_shape))
                elif var == "He":
                    self.weights = np.random.normal(0, math.sqrt(2 / in_shape), size=(in_shape, out_shape))
                elif var == "Lecun":
                    self.weights = np.random.normal(0, math.sqrt(1 / in_shape), size=(in_shape, out_shape))
                else:
                    raise ValueError("Invalid distribution")
            else:
                raise ValueError("Invalid distribution")
        else:
            self.weights = weights
        if bias is None:
            self.bias = np.zeros(out_shape)
        else:
            self.bias = bias

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass of the layer.
        |y = X * W + b|
        :param X: input values (batch size x input shape)
        :return: output values (batch size x output shape)
        """
        self.last_in = X
        return np.dot(X, self.weights) + self.bias

    def derriv(self, next_gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass of the layer.
        :param next_gradient: gradient of the next layer (batch size x output shape)
        :return: gradient of the layer (batch size x input shape)
        """
        self.gradient = np.dot(self.last_in.T, next_gradient)
        self.gradient += self.l2 * self.weights
        self.gradient += self.l1 * np.sign(self.weights)
        self.bias_gradient = next_gradient.sum(axis=0)
        return np.dot(next_gradient, self.weights.T)

    def get_gradients(self) -> dict[str, np.ndarray]:
        """
        Get the gradients of the parameters of the layer.
        Requires the backward pass to be called first.
        :return: dict[str, np.ndarray] with keys "multiplication" and "bias" for the gradients of the weights and bias
        """
        gradients = {"multiplication": self.gradient, "bias": self.bias_gradient}
        return gradients

    def get_parameters(self) -> dict[str, np.ndarray]:
        """
        Get the parameters of the layer.
        :return: dict[str, np.ndarray] with keys "multiplication" and "bias" for the weights and bias
        """
        parameters = {"multiplication": self.weights, "bias": self.bias}
        return parameters

    def change_parameters(self, changes: dict[str, np.ndarray]):
        """
        Change the parameters of the layer.
        Add the changes to the current parameters.
        :param changes: dict[str, np.ndarray] with keys "multiplication" and "bias" for the changes of the weights and bias
        """
        self.weights += changes["multiplication"]
        self.bias += changes["bias"]


class Sigmoid(Layer):
    """
    Sigmoid activation function.
    |y = 1 / (1 + e^-x)|
    """

    def __call__(self, X: np.ndarray) -> np.ndarray:
        self.last_in = X
        return 1 / (1 + np.exp(-X))

    def derriv(self, next_gradient: np.ndarray) -> np.ndarray:
        e_to_X = np.exp(self.last_in)
        return (e_to_X / (1 + e_to_X) ** 2) * next_gradient


class Tanh(Layer):
    """
    Tanh activation function.
    |y = (e^x - e^-x) / (e^x + e^-x)|
    """

    def __call__(self, X: np.ndarray) -> np.ndarray:
        self.last_in = X
        e_to_X = np.exp(X)
        e_to_minus_X = np.exp(-X)
        return (e_to_X - e_to_minus_X) / (e_to_X + e_to_minus_X)

    def derriv(self, next_gradient: np.ndarray) -> np.ndarray:
        e_to_X = np.exp(self.last_in)
        e_to_minus_X = np.exp(-self.last_in)
        return 4 / (e_to_X + e_to_minus_X) ** 2 * next_gradient


class ReLU(Layer):
    """
    Rectified linear unit activation function.
    |y = max(0, x)|
    """

    def __call__(self, X: np.ndarray) -> np.ndarray:
        self.last_in = X
        return np.where(X < 0, 0, X)

    def derriv(self, next_gradient: np.ndarray) -> np.ndarray:
        return np.where(self.last_in < 0, 0, 1) * next_gradient


class LeakyReLU(Layer):
    """
    Leaky rectified linear unit activation function.
    |y = max(alpha * x, x)|
    """

    def __init__(self, alpha: float = 0.01):
        """
        Initialize the LeakyReLU layer.
        :param alpha: alpha value of the LeakyReLU function
        """
        self.alpha = alpha

    def __call__(self, X: np.ndarray) -> np.ndarray:
        self.last_in = X
        return np.where(X < 0, self.alpha * X, X)

    def derriv(self, next_gradient: np.ndarray) -> np.ndarray:
        return np.where(self.last_in < 0, self.alpha, 1) * next_gradient


class ELU(Layer):
    """
    Exponential linear unit activation function.
    |y = max(alpha * (e^x - 1), x)|
    """

    def __init__(self, alpha: float = 1):
        """
        Initialize the ELU layer.
        :param alpha: alpha value of the ELU function
        """
        self.alpha = alpha

    def __call__(self, X: np.ndarray) -> np.ndarray:
        self.last_in = X
        return np.where(X < 0, self.alpha * (np.exp(X) - 1), X)

    def derriv(self, next_gradient: np.ndarray) -> np.ndarray:
        return np.where(self.last_in < 0, self.alpha * np.exp(self.last_in), 1) * next_gradient


class SELU(Layer):
    """
    Scaled exponential linear unit activation function.
    |y = scale * max(alpha * (e^x - 1), x)|
    used for self-normalizing neural networks
    """

    def __init__(self, alpha: float = 1.6732632423543772848170429916717,
                 scale: float = 1.0507009873554804934193349852946):
        """
        Initialize the SELU layer. (default values are used in self-normalizing neural networks)
        :param alpha: alpha value of the SELU function
        :param scale: scale value of the SELU function
        """
        self.alpha = alpha
        self.scale = scale

    def __call__(self, X: np.ndarray) -> np.ndarray:
        self.last_in = X
        return self.scale * np.where(X < 0, self.alpha * (np.exp(X) - 1), X)

    def derriv(self, next_gradient: np.ndarray) -> np.ndarray:
        return self.scale * np.where(self.last_in < 0, self.alpha * np.exp(self.last_in), 1) * next_gradient


class RReLU(Layer):
    """
    Randomized leaky rectified linear unit activation function.
    |y = max(alpha * x, x)|
    where alpha is randomly chosen from a uniform distribution between lower and upper if training
    and is the mean of lower and upper if not training.

    """

    def __init__(self, lower: float = 0.125, upper: float = 0.333):
        """
        Initialize the Randomized Rectified Linear unit layer.

        :param lower: lower bound of the uniform distribution for sampling alpha
        :param upper: upper bound of the uniform distribution for sampling alpha
        """
        self.lower = lower
        self.upper = upper
        self.train = True

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if self.train:
            self.last_alpha = np.random.uniform(self.lower, self.upper, size=X.shape)
        else:
            self.last_alpha = (self.lower + self.upper) / 2
        self.last_in = X
        return np.where(X < 0, self.last_alpha * X, X)

    def derriv(self, next_gradient: np.ndarray) -> np.ndarray:
        return np.where(self.last_in < 0, self.last_alpha, 1) * next_gradient


class PReLU(TrainableLayer):
    """
    Parametric leaky rectified linear unit activation function.
    |y = max(alpha * x, x)|
    where alpha is a trainable paramete shared across all inputs.
    alpha can be negative.
    """

    def __init__(self, alpha: float = 0.25):
        """
        Initialize the PReLU layer.
        :param alpha: initial value of alpha
        """
        self.alpha_grad = None
        self.alpha = np.array(alpha)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        self.last_in = X
        return np.where(X < 0, self.alpha * X, X)

    def derriv(self, next_gradient: np.ndarray) -> np.ndarray:
        self.alpha_grad = (np.where(self.last_in < 0, self.last_in, 0) * next_gradient).sum()
        return np.where(self.last_in < 0, self.alpha, 1) * next_gradient

    def get_gradients(self) -> dict[str, np.ndarray]:
        """
        Get the gradients of the parameters of the layer.
        :return: dict[str, np.ndarray] with keys "alpha" and values the gradients of alpha
        """
        return {"alpha": self.alpha_grad}

    def change_parameters(self, changes: dict[str, np.ndarray]):
        """
        Change the parameters of the layer.

        :param changes: dict[str, np.ndarray] with keys "alpha" and values to add to alpha
        """
        self.alpha += changes["alpha"]

    def get_parameters(self) -> dict[str, np.ndarray]:
        """
        Get the parameters of the layer.
        :return: dict[str, np.ndarray] with keys "alpha" and values the value of alpha shape (1,)
        """
        return {"alpha": self.alpha}


class Softmax(Layer):
    """
    Softmax activation function.
    |y = e^x / sum(e^x)|
    Only to be trained with CrossEntropyLoss because of implementation of derriv.
    """

    def __call__(self, X: np.ndarray) -> np.ndarray:
        self.last_in = X
        exps = np.exp(X)
        return exps / exps.sum(axis=-1, keepdims=True)

    def derriv(self, next_gradient: np.ndarray) -> np.ndarray:
        """
        The derriv of the softmax function is not implemented for the general case. (it returns copy of next_gradient)
        """
        return next_gradient


class Dropout(Layer):
    """
    Dropout layer.
    |y = x * mask|
    where mask is a binary mask with 1s with probability 1-p and 0s with probability p.
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize the Dropout layer.
        :param p: probability of dropping out a neuron
        """
        super().__init__()
        self.p = p
        self.train = True

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if self.train:
            self.last_mask = np.random.binomial(1, 1 - self.p, size=X.shape)
            return X * self.last_mask
        else:
            return X * self.p

    def derriv(self, next_gradient: np.ndarray) -> np.ndarray:
        return next_gradient * self.last_mask
