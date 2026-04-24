"""
xView GeoJSON + GeoTIFF → YOLO format converter.
Collapses xView's 60 classes into 5 ARGUS coarse classes.
"""

import json
import logging
from pathlib import Path
import numpy as np
from PIL import Image

logger = logging.getLogger("argus.preprocess")

# xView class ID → ARGUS class index
XVIEW_TO_ARGUS = {
    # tank (0)
    71: 0, 72: 0, 73: 0,
    # large_vehicle (1)
    23: 1, 32: 1, 41: 1, 42: 1, 50: 1, 53: 1,
    61: 1, 76: 1, 77: 1, 79: 1, 83: 1, 84: 1, 86: 1, 89: 1, 91: 1,
    # naval_vessel (2)
    11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 17: 2, 18: 2,
    # aircraft (3)
    21: 3, 22: 3, 24: 3, 25: 3, 26: 3,
    # structure (4)
    60: 4, 62: 4, 63: 4, 64: 4, 65: 4,
}

CLASS_NAMES = ["tank", "large_vehicle", "naval_vessel", "aircraft", "structure"]


def convert_xview_to_yolo(
    labels_path: str,
    images_dir: str,
    output_dir: str,
    chip_size: int = 640,
):
    """
    Convert xView GeoJSON labels to YOLO format txt files.

    Args:
        labels_path: path to xView_train.geojson
        images_dir:  directory containing xView GeoTIFF images
        output_dir:  where to save YOLO labels
        chip_size:   chip size used for tiling
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(labels_path) as f:
        geojson = json.load(f)

    # Group features by image filename
    by_image = {}
    for feature in geojson["features"]:
        props = feature["properties"]
        image_id = props.get("image_id", "")
        if image_id not in by_image:
            by_image[image_id] = []
        by_image[image_id].append(feature)

    logger.info(f"Found {len(by_image)} images in GeoJSON")

    converted = 0
    skipped = 0

    for image_id, features in by_image.items():
        img_path = Path(images_dir) / image_id
        if not img_path.exists():
            skipped += 1
            continue

        img = Image.open(img_path)
        img_w, img_h = img.size

        label_lines = []
        for feature in features:
            props = feature["properties"]
            type_id = props.get("type_id", -1)

            if type_id not in XVIEW_TO_ARGUS:
                continue

            class_idx = XVIEW_TO_ARGUS[type_id]
            coords = feature["geometry"]["coordinates"][0]

            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            cx = ((x_min + x_max) / 2) / img_w
            cy = ((y_min + y_max) / 2) / img_h
            w = (x_max - x_min) / img_w
            h = (y_max - y_min) / img_h

            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            w = max(0, min(1, w))
            h = max(0, min(1, h))

            label_lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if label_lines:
            label_file = output_path / (Path(image_id).stem + ".txt")
            label_file.write_text("\n".join(label_lines))
            converted += 1

    logger.info(f"Converted {converted} images, skipped {skipped}")
    return converted


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="path to xView_train.geojson")
    parser.add_argument("--images", required=True, help="directory with xView GeoTIFF images")
    parser.add_argument("--output", required=True, help="output directory for YOLO labels")
    parser.add_argument("--chip-size", type=int, default=640)
    parser.add_argument("--limit", type=int, default=0, help="process only first N images (0 = all)")
    args = parser.parse_args()
    convert_xview_to_yolo(args.labels, args.images, args.output, args.chip_size)
