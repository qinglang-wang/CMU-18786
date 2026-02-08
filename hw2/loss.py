import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    def __init__(self):
        self.diff = None

    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_gt: np.ndarray) -> float:
        ...

    @abstractmethod
    def backward(self) -> np.ndarray:
        ...


class MSELoss(Loss):
    def forward(self, y_pred, y_gt):
        self.diff = y_pred - y_gt
        return 0.5 * np.mean(self.diff**2)
    
    def backward(self):
        return self.diff / self.diff.shape[0]


class BCELoss(Loss):
    def __init__(self):
        super().__init__()
        self.y_pred, self.y_gt = None, None
        self.epsilon = 1e-9

    def forward(self, y_pred, y_gt):
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        self.y_pred, self.y_gt = y_pred, y_gt
        return -np.mean(y_gt * np.log(y_pred) + (1 - y_gt) * np.log(1 - y_pred))
    
    def backward(self):
        y_pred, y_gt = self.y_pred, self.y_gt
        return (y_pred - y_gt) / (y_pred * (1 - y_pred) + self.epsilon) / y_pred.shape[0]


class BCEFromLogitsLoss(Loss):
    def __init__(self):
        super().__init__()
        self.logits, self.y_gt = None, None

    def forward(self, logits: np.ndarray, y_gt: np.ndarray) -> float:
        self.logits, self.y_gt = logits, y_gt
        return np.mean(np.maximum(logits, 0) - logits * y_gt + np.log(1 + np.exp(-np.abs(logits))))
    
    def backward(self):
        return (1 / (1 + np.exp(-self.logits)) - self.y_gt) / self.logits.shape[0]
