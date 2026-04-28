"""
ARGUS YOLOv8 training script.
Two-phase training: frozen backbone → full fine-tune.
Run on EC2 g4dn.xlarge via ec2_train.sh.
"""

import os
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("argus.train")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="data/processed/dataset.yaml")
    parser.add_argument("--model",   default="yolov8s.pt", help="yolov8n/s/m/l/x.pt")
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--batch",   type=int, default=32)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device",  default="0", help="GPU device index, or 'cpu'")
    parser.add_argument("--freeze",  type=int, default=10, help="freeze first N backbone layers in phase 1")
    parser.add_argument("--name",    default="argus-v1")
    parser.add_argument("--s3-bucket", default=os.getenv("S3_BUCKET", ""))
    return parser.parse_args()


def upload_to_s3(local_path, bucket, key):
    import boto3
    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), bucket, key)
    logger.info(f"Uploaded {local_path} → s3://{bucket}/{key}")


def upload_run_to_s3(run_dir, bucket, s3_prefix):
    import boto3
    s3 = boto3.client("s3")
    run_dir = Path(run_dir)
    for local_file in run_dir.rglob("*"):
        if local_file.is_file():
            relative = local_file.relative_to(run_dir)
            key = f"{s3_prefix}/{relative}"
            s3.upload_file(str(local_file), bucket, key)
            logger.info(f"Uploaded {local_file} → s3://{bucket}/{key}")


def main():
    args = parse_args()
    from ultralytics import YOLO

    common = dict(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        patience=10,
        plots=True,
        save=True,
    )

    # Phase 1: frozen backbone — only detection head trains
    # Prevents destroying COCO pretrained features on early epochs
    logger.info(f"Phase 1: frozen backbone ({args.epochs // 2} epochs)")
    model = YOLO(args.model)
    model.train(
        epochs=args.epochs // 2,
        freeze=args.freeze,
        lr0=0.01,
        name=f"{args.name}-phase1",
        **common,
    )

    # Phase 2: load phase 1 best weights, unfreeze everything
    phase1_best = Path(f"runs/detect/{args.name}-phase1/weights/best.pt")
    if not phase1_best.exists():
        logger.warning(f"Phase 1 best weights not found at {phase1_best}, using last.pt")
        phase1_best = Path(f"runs/detect/{args.name}-phase1/weights/last.pt")

    logger.info(f"Phase 2: full fine-tune ({args.epochs // 2} epochs) from {phase1_best}")
    model = YOLO(str(phase1_best))
    model.train(
        epochs=args.epochs // 2,
        freeze=0,
        lr0=0.001,  # lower LR for fine-tune
        name=f"{args.name}-phase2",
        **common,
    )

    # Upload everything (weights, plots, configs, metrics) from both phases to S3
    if args.s3_bucket:
        bucket = args.s3_bucket.replace("s3://", "").split("/")[0]
        for phase in ["phase1", "phase2"]:
            run_dir = Path(f"runs/detect/{args.name}-{phase}")
            if run_dir.exists():
                upload_run_to_s3(run_dir, bucket, f"models/{args.name}/{phase}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
