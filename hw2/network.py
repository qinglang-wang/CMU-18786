import numpy as np
from abc import ABC, abstractmethod
from typing import List


class Layer(ABC):
    def __init__(self):
        self.parameters = {}
        self.state = {}
        self.cache = None
        self.is_train = True

    def __repr__(self):
        return f"{self.__class__.__name__}({self.extra_repr()})"

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def backward(self, y_grad: np.ndarray) -> np.ndarray:
        ...

    def extra_repr(self):
        return ""


class Parameter:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = np.zeros_like(data)
        
    def zero_grad(self):
        self.grad = np.zeros_like(self.data)


class Linear(Layer):
    def __init__(self, in_dim, out_dim, init_method='he'):
        super().__init__()

        self.parameters['w'] = Parameter(self._init_param(in_dim, out_dim, init_method))
        self.parameters['b'] = Parameter(np.zeros((1, out_dim)))

    @staticmethod
    def _init_param(in_dim: int, out_dim: int, init_method: str) -> np.ndarray:
        return np.random.randn(in_dim, out_dim) * (np.sqrt(2 / in_dim) if init_method == 'he' else np.sqrt(2 / (in_dim + out_dim)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x
        return x @ self.parameters['w'].data + self.parameters['b'].data

    def backward(self, y_grad: np.ndarray) -> np.ndarray:
        x = self.cache
        self.parameters['w'].grad += x.T @ y_grad
        self.parameters['b'].grad += np.sum(y_grad, axis=0, keepdims=True)
        x_grad = y_grad @ self.parameters['w'].data.T
        return x_grad
    
    def extra_repr(self):
        return f"in_features={self.parameters['w'].data.shape[0]}, out_features={self.parameters['w'].data.shape[1]}"


class ReLU(Layer):
    def forward(self, x):
        self.cache = x
        return np.maximum(x, 0)
    
    def backward(self, y_grad):
        x = self.cache
        x_grad = (x > 0).astype(float) * y_grad
        return x_grad
    

class Sigmoid(Layer):
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.cache = out
        return out
    
    def backward(self, y_grad):
        y = self.cache
        x_grad = y_grad * y * (1 - y)
        return x_grad
    

class Tanh(Layer):
    def forward(self, x):
        out = (lambda a, b: (a - b) / (a + b))(np.exp(x), np.exp(-x))
        self.cache = out
        return out
    
    def backward(self, y_grad):
        y = self.cache
        x_grad = y_grad * (1 - y**2)
        return x_grad


class Network(ABC):
    def __init__(self):
        self.layers: List[Layer] = []

    def __repr__(self):
        lines = []

        lines.append(f"{self.__class__.__name__}(")
        for i, layer in enumerate(self.layers):
            lines.append(f"  ({i}): {repr(layer)}")

        lines.append(")")
        return "\n".join(lines)


class MLP(Network):
    def __init__(self, layer_nodes: List, optimizer, hidden_act: str='relu', output_act: str=None, init_method: str='he'):
        super().__init__()
        self.optimizer = optimizer

        self.layers.append(Linear(layer_nodes[0], layer_nodes[1], init_method=init_method))
        for i in range(1, len(layer_nodes) - 1):
            if hidden_act.lower() != 'linear':
                self.layers.append(self._get_activation(hidden_act))
            self.layers.append(Linear(layer_nodes[i], layer_nodes[i+1], init_method=init_method))
        if output_act:
            self.layers.append(self._get_activation(output_act))

    def _get_activation(self, act: str) -> Layer:
        act = act.lower()
        if act == 'relu':
            return ReLU()
        elif act == 'sigmoid':
            return Sigmoid()
        elif act == 'tanh':
            return Tanh()
        else:
            raise ValueError("Activation function not implemented.")

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, out_grad: np.ndarray):
        grad = out_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def zero_grad(self):
        for layer in self.layers:
            if layer.parameters:
                for param in layer.parameters.values():
                    param.zero_grad()

    def step(self):
        for layer in self.layers:
            if layer.parameters:
                self.optimizer.update(layer)
