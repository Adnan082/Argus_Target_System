"""
ARGUS YOLOv8x training script.
Run on EC2 p3.2xlarge with data in S3.
"""

import os
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("argus.train")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="configs/dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--freeze", type=int, default=10, help="freeze first N layers")
    parser.add_argument("--model", default="yolov8x.pt")
    parser.add_argument("--name", default="argus-v1")
    parser.add_argument("--s3-bucket", default=os.getenv("S3_BUCKET", ""))
    return parser.parse_args()


def upload_model_to_s3(local_path: str, bucket: str, key: str):
    import boto3
    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket, key)
    logger.info(f"Uploaded {local_path} to s3://{bucket}/{key}")


def main():
    args = parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)

    logger.info(f"Starting training: {args.epochs} epochs, batch={args.batch}")

    # Phase 1: frozen backbone
    logger.info("Phase 1: Training with frozen backbone")
    model.train(
        data=args.data,
        epochs=args.epochs // 2,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=args.freeze,
        name=f"{args.name}-phase1",
        patience=20,
        save=True,
    )

    # Phase 2: full fine-tune
    logger.info("Phase 2: Full fine-tune")
    model.train(
        data=args.data,
        epochs=args.epochs // 2,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=0,
        name=f"{args.name}-phase2",
        patience=20,
        save=True,
    )

    # Upload best weights to S3
    best_weights = f"runs/detect/{args.name}-phase2/weights/best.pt"
    if args.s3_bucket and Path(best_weights).exists():
        upload_model_to_s3(best_weights, args.s3_bucket, f"models/{args.name}/best.pt")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
