"""
RunPod Serverless Handler for YOLOv8 inference.
Deploy as a Docker container to RunPod.
"""

import base64
import io
from PIL import Image


def handler(event):
    """Process a batch of chips through YOLOv8."""
    import runpod
    from ultralytics import YOLO

    model = YOLO("/model/best.pt")

    input_data = event["input"]
    chips_b64 = input_data.get("chips", [])

    all_detections = []
    for i, chip_b64 in enumerate(chips_b64):
        img_bytes = base64.b64decode(chip_b64)
        img = Image.open(io.BytesIO(img_bytes))
        results = model(img, conf=0.25, iou=0.45, imgsz=640)

        for r in results:
            for box in r.boxes:
                all_detections.append({
                    "chip_index": i,
                    "class_id": int(box.cls),
                    "class_name": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxyn.tolist()[0],
                })

    return {"detections": all_detections, "chips_processed": len(chips_b64)}


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
