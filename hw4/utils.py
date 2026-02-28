import os
import random
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from data import CIFAR100_CLASSES

matplotlib.use('Agg')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def plot_loss(history, title='', save_to=None):
    """Plot train and val loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], label='Train Loss', lw=2)
    ax.plot(epochs, history['val_loss'],   label='Val Loss',   lw=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Loss Curve {title}')
    ax.legend()
    ax.grid(True)
    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        fig.savefig(save_to, bbox_inches='tight', dpi=150)
    plt.close(fig)

def plot_accuracy(history, title='', save_to=None):
    """Plot train and val accuracy curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history['train_acc']) + 1)
    ax.plot(epochs, history['train_acc'], label='Train Acc', lw=2)
    ax.plot(epochs, history['val_acc'],   label='Val Acc',   lw=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Accuracy Curve {title}')
    ax.legend()
    ax.grid(True)
    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        fig.savefig(save_to, bbox_inches='tight', dpi=150)
    plt.close(fig)

def visualize_predictions(model, dataset, indices, title='', save_to=None, device='cuda'):
    """Show images with ground truth and predicted labels."""
    model.eval()

    mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
    std = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)

    n = len(indices)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4))
    if n == 1:
        axes = [axes]

    with torch.no_grad():
        for ax, idx in zip(axes, indices):
            img, label = dataset[idx]
            logits = model(img.unsqueeze(0).to(device))
            pred = logits.argmax(dim=1).item()

            # Reverse normalization for display
            img_show = img * std + mean
            img_show = img_show.clamp(0, 1).permute(1, 2, 0).numpy()

            ax.imshow(img_show)
            ax.set_title(f"GT: {CIFAR100_CLASSES[label]}\nPred: {CIFAR100_CLASSES[pred]}", fontsize=9)
            ax.axis('off')

    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        fig.savefig(save_to, bbox_inches='tight', dpi=150)
    plt.close(fig)
