"""
Run inference on all images in a directory and print a summary.
Usage: python scripts/batch_inference.py --dir data/processed/images/val/
"""

import argparse
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO


def run(image_dir, model_path, conf, save):
    model = YOLO(model_path)
    images = list(Path(image_dir).glob("*.jpg"))
    print(f"Running inference on {len(images)} images...\n")

    class_counts = defaultdict(int)
    total_detections = 0
    images_with_detections = 0

    for img_path in images:
        results = model(str(img_path), conf=conf, iou=0.45, imgsz=640, verbose=False)
        r = results[0]

        if len(r.boxes) > 0:
            images_with_detections += 1
            total_detections += len(r.boxes)
            for box in r.boxes:
                cls = model.names[int(box.cls)]
                class_counts[cls] += 1
            if save:
                r.save(filename=f"detections/{img_path.stem}_det.jpg")

    print("=" * 40)
    print(f"Images processed:        {len(images)}")
    print(f"Images with detections:  {images_with_detections}")
    print(f"Total detections:        {total_detections}")
    print(f"Avg detections/image:    {total_detections/len(images):.1f}")
    print("\nDetections by class:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:15s}  {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",   default="data/processed/images/val/")
    parser.add_argument("--model", default="models/argus-v1/phase2/weights/best.pt")
    parser.add_argument("--conf",  type=float, default=0.25)
    parser.add_argument("--save",  action="store_true", help="Save annotated images to detections/")
    args = parser.parse_args()

    if args.save:
        Path("detections").mkdir(exist_ok=True)

    run(args.dir, args.model, args.conf, args.save)
