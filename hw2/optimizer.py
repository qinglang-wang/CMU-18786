import numpy as np
from abc import ABC, abstractmethod
from network import Layer


class Optimizer(ABC):
    def __init__(self, lr: float=1e-3):
        self.lr = lr

    def __repr__(self):
        return f"{self.__class__.__name__}(lr={self.lr})"
    
    @abstractmethod
    def update(self, layer: Layer):
        ...
    

class SGD(Optimizer):
    def __init__(self, lr=1e-3):
        super().__init__(lr)

    def update(self, layer):
        if hasattr(layer, 'parameters'):
            for param in layer.parameters.values():
                param.data -= self.lr * param.grad


class SGDWithMomentum(Optimizer):
    def __init__(self, lr=1e-3, momentum: float=0.9):
        super().__init__(lr)
        self.velocity = {}
        self.momentum = momentum
    
    def __repr__(self):
        return f"{self.__class__.__name__}(lr={self.lr}, momentum={self.momentum})"

    def update(self, layer):
        if hasattr(layer, 'parameters'):
            for param in layer.parameters.values():
                if param not in self.velocity:
                    self.velocity[param] = np.zeros_like(param.data)
                
                self.velocity[param] = self.momentum * self.velocity[param] + param.grad
                param.data -= self.lr * self.velocity[param]


class Adam(Optimizer):
    def __init__(self, lr=1e-3, beta1: float=0.9, beta2: float=0.999, epsilon: float=1e-9):
        super().__init__(lr)
        self.state = {}
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def __repr__(self):
        return f"{self.__class__.__name__}(lr={self.lr}, betas=({self.beta1}, {self.beta2}), eps={self.epsilon})"
    
    def update(self, layer):
        if hasattr(layer, 'parameters'):
            for param in layer.parameters.values():
                if param not in self.state:
                    self.state[param] = {
                        'm': np.zeros_like(param.data),
                        'v': np.zeros_like(param.data),
                        't': 0
                    }
                state = self.state[param]
                
                state['t'] += 1
                state['m'] = self.beta1 * state['m'] + (1 - self.beta1) * param.grad
                state['v'] = self.beta2 * state['v'] + (1 - self.beta2) * param.grad ** 2

                m_hat = state['m'] / (1 - self.beta1 ** state['t'])
                v_hat = state['v'] / (1 - self.beta2 ** state['t'])

                param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
