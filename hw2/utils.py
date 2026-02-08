import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict

def count_binary_correct(y_pred: np.ndarray, y_gt: np.ndarray, threshold: int=0.5) -> int:
    return np.sum((y_pred >= threshold).astype(float) == y_gt)

def count_binary_correct_from_logits(logits: np.ndarray, y_gt: np.ndarray) -> int:
    return np.sum((logits > 0).astype(float) == y_gt)

def set_global_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)

def print_config(name: str, config: Dict, indentation: int=2):
    print(f"{' ' * indentation}{name}:")
    for k, v in config.items():
        if isinstance(v, np.ndarray):
            print(f"{' ' * indentation}  {k}: <ndarray shape={v.shape}>")
        else:
            print(f"{' ' * indentation}  {k}: {v}")

def plot_loss(ax, logs):
    """
    Function to plot training and validation/test loss curves
    :param logs: dict with keys 'train_loss','val_loss' and 'epochs', where train_loss and val_loss are lists with 
                the training and test/validation loss for each epoch
    """
    t = np.arange(len(logs['train_loss']))
    ax.plot(t, logs['train_loss'], label='train_loss', lw=3)
    ax.plot(t, logs['val_loss'], label='val_loss', lw=3)
    ax.grid(True)
    ax.set_xlabel('epochs', fontsize=15)
    ax.set_ylabel('loss value', fontsize=15)
    ax.legend(fontsize=15)
    ax.set_title('Loss Curve', fontsize=15)

def plot_decision_boundary(ax, X, y, model, pred_fn, boundry_level=None):
    """
    Plots the decision boundary for the model prediction
    :param X: input data
    :param y: true labels
    :param pred_fn: prediction function,  which use the current model to predict。. i.e. y_pred = pred_fn(X)
    :boundry_level: Determines the number and positions of the contour lines / regions.
    :return:
    """
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    Z = pred_fn(model, np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.7, levels=boundry_level, cmap='viridis_r')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.scatter(X[:, 0], X[:, 1], c=y.reshape(-1), alpha=0.7,s=50, cmap='viridis_r',)

def visualize(model, history, valset_config, pred_fn: Callable = lambda model, x: model.forward(x), save_to: str=None, display: bool=True):
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    plot_loss(axes[0], history)
    plot_decision_boundary(axes[1], valset_config['data'], valset_config['labels'], model, pred_fn=pred_fn)
    axes[1].set_title(f"Decision Boundary | Iteration: {len(history['train_loss'])} | Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {history['val_loss'][-1]:.4f} | Best Val Acc: {np.max(history['val_acc']):.4f}", fontsize=14)
    
    if save_to:
        fig.savefig(save_to, bbox_inches='tight', dpi=150)
    if display:
        plt.tight_layout()
        plt.show()
