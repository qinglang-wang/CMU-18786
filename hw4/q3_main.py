import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as tv_models
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from typing import List, Tuple
from utils import set_seed

matplotlib.use('Agg')
torch.set_float32_matmul_precision('high')

class NaiveObjectDetector:
    # ImageNet class indices for cats and dogs
    CAT_INDICES = list(range(281, 286))  # cat: 281-285
    DOG_INDICES = list(range(151, 269))  # dog: 151-268

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self._load_classifier()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _load_classifier(self):
        """Load a pretrained ResNet18 (from torchvision) for ImageNet classification."""
        model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1).to(self.device).eval()
        return model

    def classify_patch(self, patch_img: Image.Image, threshold: float=0.3) -> Tuple:
        """
        Classify a PIL image patch.
        Returns:
            label: str if cat or dog, else None
            confidence: float
        """
        x = self.transform(patch_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).squeeze()

        cat_probs = probs[self.CAT_INDICES]
        dog_probs = probs[self.DOG_INDICES]

        if max(cat_probs.sum(), dog_probs.sum()) < threshold:
            return None, 0.0

        return max([('cat', cat_probs.max().item()), ('dog', dog_probs.max().item())], key=lambda x: x[1])

    def baseline_detection(self, image: Image.Image, grid_size: int=5, threshold: float=0.15) -> List:
        """
        Divide image into grid_size x grid_size non-overlapping patches and classify each.
        Returns:
            List (x1, y1, x2, y2, label, confidence)
        """
        W, H = image.size
        pw, ph = W // grid_size, H // grid_size
        boxes = []

        for i in range(grid_size):
            for j in range(grid_size):
                x1, y1 = j * pw, i * ph
                x2, y2 = x1 + pw, y1 + ph
                patch = image.crop((x1, y1, x2, y2))
                label, conf = self.classify_patch(patch, threshold)
                if label:
                    boxes.append((x1, y1, x2, y2, label, conf))
        return boxes

    def sliding_window_detection(self, image: Image.Image, window_sizes: List[int]=[64, 96, 128, 160, 192], stride_ratio: float=0.5, threshold: float=0.2) -> List:
        """
        Multi-scale sliding window detection with varying window sizes.
        Returns:
            List (x1, y1, x2, y2, label, confidence)
        """
        W, H = image.size
        boxes = []

        for window_size in window_sizes:
            stride = max(int(window_size * stride_ratio), 16)
            for y in range(0, H - window_size + 1, stride):
                for x in range(0, W - window_size + 1, stride):
                    patch = image.crop((x, y, x + window_size, y + window_size))
                    label, conf = self.classify_patch(patch, threshold)
                    if label:
                        boxes.append((x, y, x + window_size, y + window_size, label, conf))
        return boxes

    @staticmethod
    def compute_iou(box1: Tuple, box2: Tuple) -> float:
        """Compute IoU between two boxes in (x, y, x, y)."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    @staticmethod
    def nms(boxes: List, iou_threshold: float=0.4) -> List:
        """Non-Maximum Suppression on boxes in (x, y, x, y, label, confidence)."""
        if not boxes:
            return []

        # Sort by descending confidence
        boxes = sorted(boxes, key=lambda b: b[5], reverse=True)
        keep = []

        while boxes:
            best = boxes.pop(0)
            keep.append(best)
            boxes = [b for b in boxes if b[4] != best[4] or NaiveObjectDetector.compute_iou(best[:4], b[:4]) < iou_threshold]

        return keep

    @staticmethod
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


def main():
    set_seed(42)
    detector = NaiveObjectDetector()

    image_files = ['2007_001239.jpg', '2008_002152.jpg']
    os.makedirs('results', exist_ok=True)

    print("Experiment: Naive Object Detection")
    for image_file in image_files:
        print(f"Detecting objects in {image_file}:")
        image = Image.open(image_file).convert('RGB')
        image_name = os.path.splitext(image_file)[0]

        # Baseline: 5x5 grid
        baseline_boxes = detector.baseline_detection(image, threshold=0.0)
        detector.draw_boxes(image, baseline_boxes, title=f'Baseline 5x5 Grid - {image_file}', save_to=f'results/q3_baseline_{image_name}.png')

        # Improved: multi-scale sliding window + NMS
        raw_boxes = detector.sliding_window_detection(image, window_sizes=[64, 96, 128, 160, 192], stride_ratio=0.5, threshold=0.2)
        nms_boxes = detector.nms(raw_boxes, iou_threshold=0.3)
        detector.draw_boxes(image, nms_boxes, title=f'Multi-Scale + NMS - {image_file}', save_to=f'results/q3_improved_{image_name}.png')

if __name__ == '__main__':
    main()
