import os
import json
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from collections import defaultdict

CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]


class PreloadedCIFAR100(Dataset):
    """
    Loads the entire CIFAR100 dataset into RAM at initialization to eliminate disk I/O.
    Images are stored as numpy arrays and converted to PIL Image on the fly for torchvision transforms.
    """
    def __init__(self, root='./data', train=True, transform=None):
        self.dataset = torchvision.datasets.CIFAR100(root=root, train=train, download=True)
        self.transform = transform

        self.data = self.dataset.data
        self.targets = self.dataset.targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target


def get_cifar100_loaders(batch_size=128, augment=False, num_workers=4):
    """
    Load CIFAR-100 dataset with optional augmentation.

    [output]
    * train_loader: DataLoader for training set
    * val_loader: DataLoader for validation set (original test split)
    """
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = PreloadedCIFAR100(root='./data', train=True, transform=train_transform)
    val_dataset = PreloadedCIFAR100(root='./data', train=False, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        num_workers=num_workers
    )

    return train_loader, val_loader

# YOLOv8 uses 0-79 class indices; COCO uses specific category IDs.
# This maps YOLO index -> COCO category ID.
YOLO_TO_COCO = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90
]

# COCO 80 class names (in order matching YOLO indices 0-79)
COCO_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

def load_coco_annotations(coco_dir='data'):
    """
    Load COCO 2017 val annotations.

    [output]
    * gt_by_image: dict mapping image_id -> list of ground truth entries
    * images_info: dict mapping image_id -> image metadata
    * cat_id_to_name: dict mapping category_id -> category name
    * all_gts: flat list of all ground truth entries
    """
    ann_file = os.path.join(coco_dir, 'annotations', 'instances_val2017.json')
    with open(ann_file, 'r') as f:
        coco = json.load(f)

    images = {img['id']: img for img in coco['images']}
    cat_id_to_name = {c['id']: c['name'] for c in coco['categories']}

    gt_by_image = defaultdict(list)
    all_ground_truths = []
    for ann in coco['annotations']:
        if ann.get('iscrowd', 0):
            continue
        x, y, w, h = ann['bbox']
        bbox = [x, y, x + w, y + h]
        gt_entry = {'image_id': ann['image_id'], 'category_id': ann['category_id'], 'bbox': bbox}
        gt_by_image[ann['image_id']].append(gt_entry)
        all_ground_truths.append(gt_entry)

    return gt_by_image, images, cat_id_to_name, all_ground_truths
