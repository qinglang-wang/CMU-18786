import os
import random
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image
from typing import List
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

def compute_iou(box1, box2):
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def nms(boxes, iou_threshold=0.4):
    """
    Non-Maximum Suppression on boxes in (x1, y1, x2, y2, label, confidence).
    Returns filtered list of boxes.
    """
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[5], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes if b[4] != best[4] or compute_iou(best[:4], b[:4]) < iou_threshold]
    return keep

def compute_ap(precision_list, recall_list):
    """Compute Average Precision using 11-point interpolation."""
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        precisions_at_t = [p for p, r in zip(precision_list, recall_list) if r >= t]
        ap += max(precisions_at_t) if precisions_at_t else 0.0
    return ap / 11.0

def compute_map50(all_predictions, all_ground_truths, iou_threshold=0.5):
    """Compute mAP@50 across all classes."""
    from collections import defaultdict

    gt_by_img_cat = defaultdict(list)
    for gt in all_ground_truths:
        gt_by_img_cat[(gt['image_id'], gt['category_id'])].append(gt['bbox'])

    all_categories = set(gt['category_id'] for gt in all_ground_truths)
    preds_by_cat = defaultdict(list)
    for pred in all_predictions:
        preds_by_cat[pred['category_id']].append(pred)

    aps = []
    for cat_id in all_categories:
        cat_preds = sorted(preds_by_cat.get(cat_id, []), key=lambda x: x['score'], reverse=True)
        n_gt = sum(len(v) for (img_id, c), v in gt_by_img_cat.items() if c == cat_id)
        if n_gt == 0:
            continue

        matched = {}
        for (img_id, c), boxes in gt_by_img_cat.items():
            if c == cat_id:
                matched[(img_id, c)] = [False] * len(boxes)

        tp_list, fp_list = [], []
        for pred in cat_preds:
            img_id = pred['image_id']
            pred_box = pred['bbox']
            gt_boxes = gt_by_img_cat.get((img_id, cat_id), [])
            gt_matched = matched.get((img_id, cat_id), [])

            best_iou, best_idx = 0.0, -1
            for idx, gt_box in enumerate(gt_boxes):
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= iou_threshold and best_idx >= 0 and not gt_matched[best_idx]:
                tp_list.append(1)
                fp_list.append(0)
                gt_matched[best_idx] = True
            else:
                tp_list.append(0)
                fp_list.append(1)

        tp_cumsum = np.cumsum(tp_list)
        fp_cumsum = np.cumsum(fp_list)
        recall = tp_cumsum / n_gt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)

        aps.append(compute_ap(precision.tolist(), recall.tolist()))

    return np.mean(aps) if aps else 0.0

def evaluate_detector(detector, image_ids, images_info, img_dir):
    """Run a detector over all images and collect predictions."""
    all_preds = []
    for img_id in image_ids:
        img_path = os.path.join(img_dir, images_info[img_id]['file_name'])
        preds = detector.predict(img_path)
        for p in preds:
            p['image_id'] = img_id
        all_preds.extend(preds)
    return all_preds

def measure_latency(predict_fn, image_ids, images_info, img_dir, num_images=100):
    """Measure average per-image inference latency (seconds) for a predict function."""
    import time
    sample_ids = random.sample(image_ids, min(num_images, len(image_ids)))
    times = []
    for img_id in sample_ids:
        img_path = os.path.join(img_dir, images_info[img_id]['file_name'])
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        predict_fn(img_path)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append(time.perf_counter() - t0)
    return np.mean(times)

def draw_boxes(image: Image.Image, boxes: List, title: str='', save_to: str=None):
    """Draw bounding boxes on image."""
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(np.array(image))

    colors = {'cat': 'lime', 'dog': 'deepskyblue'}
    for x1, y1, x2, y2, label, conf in boxes:
        color = colors.get(label, 'red')
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 4, f'{label} {conf:.2f}', color=color, fontsize=10, fontweight='bold', backgroundcolor='black')

    ax.set_title(title, fontsize=14)
    ax.axis('off')
    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        fig.savefig(save_to, bbox_inches='tight', dpi=150)
    plt.close(fig)