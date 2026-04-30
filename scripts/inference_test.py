"""
Quick local inference test — run on any image to see ARGUS detections.
Usage: python scripts/inference_test.py --image path/to/image.jpg
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def run(image_path, model_path, conf, show):
    model = YOLO(model_path)
    results = model(image_path, conf=conf, iou=0.45, imgsz=640)

    for r in results:
        print(f"\nImage: {image_path}")
        print(f"Detections: {len(r.boxes)}")
        for box in r.boxes:
            cls = model.names[int(box.cls)]
            conf_score = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"  {cls:15s}  conf={conf_score:.2f}  box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

        if show:
            r.save(filename=f"detection_{Path(image_path).stem}.jpg")
            print(f"\nSaved annotated image: detection_{Path(image_path).stem}.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  required=True, help="Path to input image")
    parser.add_argument("--model",  default="models/argus-v1/phase2/weights/best.pt")
    parser.add_argument("--conf",   type=float, default=0.25)
    parser.add_argument("--save",   action="store_true", help="Save annotated image")
    args = parser.parse_args()

    run(args.image, args.model, args.conf, args.save)
