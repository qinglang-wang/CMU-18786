import os
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
from data import YOLO_TO_COCO, load_coco_annotations
from utils import set_seed, compute_map50, evaluate_detector, measure_latency


class FasterRCNNDetector:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1).to(device).eval()
        self.transform = transforms.ToTensor()

    def predict(self, img_path):
        """Run inference on a single image, return list of prediction dicts."""
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img).to(self.device)

        with torch.no_grad():
            outputs = self.model([img_tensor])[0]

        return [
            {
                'bbox': outputs['boxes'][i].cpu().tolist(),
                'score': outputs['scores'][i].cpu().item(),
                'category_id': outputs['labels'][i].cpu().item(),
            }
            for i in range(len(outputs['boxes']))
        ]


class YOLODetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)

    def predict(self, img_path):
        """Run inference on a single image, return list of prediction dicts."""
        results = self.model(img_path, verbose=False)
        preds = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                yolo_cls = int(boxes.cls[i].cpu().item())
                cat_id = YOLO_TO_COCO[yolo_cls] if yolo_cls < len(YOLO_TO_COCO) else yolo_cls
                preds.append(
                    {
                        'bbox': boxes.xyxy[i].cpu().tolist(),
                        'score': boxes.conf[i].cpu().item(),
                        'category_id': cat_id,
                    }
                )
        return preds


def main():
    set_seed(42)
    coco_dir = 'data'
    img_dir = os.path.join(coco_dir, 'val2017')
    gt_by_image, images_info, cat_id_to_name, all_gts = load_coco_annotations(coco_dir)
    all_image_ids = list(images_info.keys())

    print("Experiment: Object Detection Performance Profiling")
    frcnn = FasterRCNNDetector()
    frcnn_preds = evaluate_detector(frcnn, all_image_ids, images_info, img_dir)
    frcnn_map50 = compute_map50(frcnn_preds, all_gts)

    yolo8 = YOLODetector('ckpts/yolov8n.pt')
    yolo8_preds = evaluate_detector(yolo8, all_image_ids, images_info, img_dir)
    yolo8_map50 = compute_map50(yolo8_preds, all_gts)

    frcnn_latency = measure_latency(frcnn.predict, all_image_ids, images_info, img_dir)
    yolo8_latency = measure_latency(yolo8.predict, all_image_ids, images_info, img_dir)

    print(f"{'='*50}")
    print(f"{'Model':<20} {'mAP50':>10} {'Latency (ms)':>15}")
    print(f"{'-'*50}")
    print(f"{'Faster-RCNN':<20} {frcnn_map50:>10.4f} {frcnn_latency*1000:>15.2f}")
    print(f"{'YOLOv8n':<20} {yolo8_map50:>10.4f} {yolo8_latency*1000:>15.2f}")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()
