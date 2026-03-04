import os
import tqdm
from PIL import Image
from ultralytics import YOLO
from data import YOLO_TO_COCO, COCO_CLASS_NAMES, load_coco_annotations
from utils import set_seed, compute_map50, draw_boxes

def evaluate_world_map50(coco_dir: str) -> float:
    """Evaluate YOLOv8s-world mAP50 on COCO Val using COCO class prompts."""
    img_dir = os.path.join(coco_dir, 'val2017')

    model = YOLO('ckpts/yolov8s-world.pt')
    model.set_classes(COCO_CLASS_NAMES)

    gt_by_image, images_info, cat_id_to_name, all_gts = load_coco_annotations(coco_dir)

    print("Evaluating YOLOv8s-world mAP50 on COCO Val...")
    all_preds = []
    for img_id, info in tqdm.tqdm(images_info.items(), desc="Evaluating"):
        img_path = os.path.join(img_dir, info['file_name'])
        results = model(img_path, verbose=False)
        for r in results:
            for i in range(len(r.boxes)):
                yolo_cls = int(r.boxes.cls[i].cpu().item())
                coco_cat_id = YOLO_TO_COCO[yolo_cls] if yolo_cls < len(YOLO_TO_COCO) else yolo_cls
                all_preds.append(
                    {
                        'image_id': img_id,
                        'category_id': coco_cat_id,
                        'bbox': r.boxes.xyxy[i].cpu().tolist(),
                        'score': r.boxes.conf[i].cpu().item(),
                    }
                )

    map50 = compute_map50(all_preds, all_gts)
    print(f"  YOLOv8s-world achieved {map50:.4f} mAP50 on COCO Val.")
    return map50


def detect_catdog(image_files: list, save_dir: str) -> None:
    """Run cat/dog open-vocabulary detection and save visualizations."""
    class_names = ['cat', 'dog']

    model = YOLO('ckpts/yolov8s-world.pt')
    model.set_classes(class_names)

    print("\nDetecting cats and dogs...")
    results = []
    for image_file in tqdm.tqdm(image_files, desc="Detecting"):
        image = Image.open(image_file).convert('RGB')
        predictions = model(image_file, verbose=False)

        boxes = []
        for r in predictions:
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().tolist()
                label = class_names[int(r.boxes.cls[i].cpu().item())]
                conf = r.boxes.conf[i].cpu().item()
                boxes.append((x1, y1, x2, y2, label, conf))

        results.append(boxes)
        image_name = os.path.splitext(image_file)[0]
        draw_boxes(image, boxes, title=f'YOLOv8s-world Cat/Dog — {image_file}', save_to=f'{save_dir}/q5_catdog_{image_name}.png')
    return results

def main():
    set_seed(42)
    image_files=['2007_001239.jpg', '2008_002152.jpg']

    print("\nExperiment: Open Vocabulary Object Detection")

    # Part 1: mAP50 on COCO Val
    map50 = evaluate_world_map50(coco_dir='data')

    # Part 2: Cat/dog detection on provided images
    results = detect_catdog(image_files=image_files, save_dir='results')

    print(f"{'='*35}")
    print(f"{'Model':<20} {'mAP50':>10}")
    print(f"{'-'*35}")
    print(f"{'YOLOv8s-world':<20} {map50:>10.4f}")
    print(f"{'='*35}")
    print(f"{'Image':<20} {'cat':>5} {'dog':>5}")
    print(f"{'-'*35}")
    for image_file, boxes in zip(image_files, results):
        cat_count = sum(1 for _, _, _, _, label, _ in boxes if label == 'cat')
        dog_count = sum(1 for _, _, _, _, label, _ in boxes if label == 'dog')
        print(f"{os.path.basename(image_file):<20} {cat_count:>5} {dog_count:>5}")
    print(f"{'='*35}")

if __name__ == '__main__':
    main()
