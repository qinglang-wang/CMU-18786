import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as tv_models
import matplotlib
from PIL import Image
from typing import List, Tuple
from utils import set_seed, draw_boxes, nms

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
            logits = self.model(x).squeeze(0)
            probs = F.softmax(logits, dim=0)

        cat_probs = probs[self.CAT_INDICES]
        dog_probs = probs[self.DOG_INDICES]

        return max([('cat', cat_probs.max().item()), ('dog', dog_probs.max().item()), (None, threshold)], key=lambda x: x[1])

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

    def sliding_window_detection(self, image: Image.Image, window_sizes: List[int]=[64, 96, 128, 160, 192], aspect_ratios: List[float]=[0.5, 1.0, 2.0], stride_ratio: float=0.2, threshold: float=0.5) -> List:
        """
        Multi-scale sliding window detection with varying window sizes and aspect ratios.
        Returns:
            List (x1, y1, x2, y2, label, confidence)
        """
        W, H = image.size
        boxes = []

        for window_size in window_sizes:
            for ar in aspect_ratios:
                w = int(window_size * (ar ** 0.5))
                h = int(window_size / (ar ** 0.5))

                if w > W or h > H:
                    continue

                stride_x = max(int(w * stride_ratio), 4)
                stride_y = max(int(h * stride_ratio), 4)

                for y in range(0, H - h + 1, stride_y):
                    for x in range(0, W - w + 1, stride_x):
                        patch = image.crop((x, y, x + w, y + h))
                        label, conf = self.classify_patch(patch, threshold)
                        if label:
                            boxes.append((x, y, x + w, y + h, label, conf))
        return boxes


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
        baseline_boxes = detector.baseline_detection(image, threshold=0.05)
        draw_boxes(image, baseline_boxes, title=f'Baseline 5x5 Grid - {image_file}', save_to=f'results/q3_baseline_{image_name}.png')

        # Improved: multi-scale sliding window + NMS
        raw_boxes = detector.sliding_window_detection(image, window_sizes=[64, 96, 128, 160, 192, 224], aspect_ratios=[0.5, 0.75, 1.0, 1.5, 2.0], stride_ratio=0.10, threshold=0.6)
        nms_boxes = nms(raw_boxes, iou_threshold=0.01)
        draw_boxes(image, nms_boxes, title=f'Multi-Scale + NMS - {image_file}', save_to=f'results/q3_improved_{image_name}.png')

if __name__ == '__main__':
    main()
