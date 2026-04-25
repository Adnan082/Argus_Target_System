"""
xView → YOLO chips pipeline.

  1. Convert GeoJSON → per-image YOLO labels (using bounds_imcoords pixel coords)
  2. Stratified 80/20 train/val split at image level, by rarest ARGUS class present
  3. Chip each image into 640x640 tiles with 50% overlap
  4. Write chips to data/processed/{images,labels}/{train,val}/
  5. Write dataset.yaml
"""

import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("argus.chip")

CHIP_SIZE = 640
STRIDE = 320  # 50% overlap
VAL_RATIO = 0.2
SEED = 42

# structure=4 is rarest, tank=0 is most common
RARITY_ORDER = [4, 3, 1, 2, 0]


def load_by_image(labels_path):
    with open(labels_path) as f:
        geojson = json.load(f)
    by_image = defaultdict(list)
    for feature in geojson["features"]:
        img_id = feature["properties"].get("image_id", "")
        by_image[img_id].append(feature)
    return by_image


def image_classes(img_id, by_image, xview_to_argus):
    classes = set()
    for f in by_image.get(img_id, []):
        tid = f["properties"].get("type_id", -1)
        if tid in xview_to_argus:
            classes.add(xview_to_argus[tid])
    return classes


def stratified_split(image_ids, by_image, xview_to_argus):
    random.seed(SEED)
    by_label = defaultdict(list)
    for img_id in image_ids:
        classes = image_classes(img_id, by_image, xview_to_argus)
        label = next((c for c in RARITY_ORDER if c in classes), -1)
        by_label[label].append(img_id)

    train_ids, val_ids = [], []
    for label in sorted(by_label):
        ids = by_label[label]
        random.shuffle(ids)
        n_val = max(1, round(len(ids) * VAL_RATIO))
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])
        log.info(f"  class_label={label}: {len(ids)} images → {len(ids)-n_val} train / {n_val} val")

    return train_ids, val_ids


def chip_image(img_path, label_path, out_img_dir, out_lbl_dir):
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size
    img_arr = np.array(img)

    boxes = []
    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            if not line:
                continue
            parts = line.split()
            cls = int(parts[0])
            cx_px = float(parts[1]) * img_w
            cy_px = float(parts[2]) * img_h
            bw_px = float(parts[3]) * img_w
            bh_px = float(parts[4]) * img_h
            boxes.append((cls, cx_px, cy_px, bw_px, bh_px))

    stem = img_path.stem
    chips_written = 0

    xs = list(range(0, max(1, img_w - CHIP_SIZE + 1), STRIDE))
    ys = list(range(0, max(1, img_h - CHIP_SIZE + 1), STRIDE))
    if not xs or xs[-1] + CHIP_SIZE < img_w:
        xs.append(max(0, img_w - CHIP_SIZE))
    if not ys or ys[-1] + CHIP_SIZE < img_h:
        ys.append(max(0, img_h - CHIP_SIZE))
    xs = sorted(set(xs))
    ys = sorted(set(ys))

    for y in ys:
        for x in xs:
            chip_boxes = []
            for cls, cx_px, cy_px, bw_px, bh_px in boxes:
                if x <= cx_px < x + CHIP_SIZE and y <= cy_px < y + CHIP_SIZE:
                    new_cx = max(0.0, min(1.0, (cx_px - x) / CHIP_SIZE))
                    new_cy = max(0.0, min(1.0, (cy_px - y) / CHIP_SIZE))
                    new_w  = min(bw_px / CHIP_SIZE, 1.0)
                    new_h  = min(bh_px / CHIP_SIZE, 1.0)
                    chip_boxes.append(f"{cls} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}")

            # Extract chip, pad if edge chip is smaller than CHIP_SIZE
            chip_arr = img_arr[y:y + CHIP_SIZE, x:x + CHIP_SIZE]
            if chip_arr.shape[0] < CHIP_SIZE or chip_arr.shape[1] < CHIP_SIZE:
                padded = np.zeros((CHIP_SIZE, CHIP_SIZE, 3), dtype=np.uint8)
                padded[:chip_arr.shape[0], :chip_arr.shape[1]] = chip_arr
                chip_arr = padded

            name = f"{stem}_{x}_{y}"
            Image.fromarray(chip_arr).save(out_img_dir / f"{name}.jpg", quality=95)
            (out_lbl_dir / f"{name}.txt").write_text("\n".join(chip_boxes))
            chips_written += 1

    return chips_written


def main(labels_path, images_dir, output_dir):
    labels_path = Path(labels_path)
    images_dir  = Path(images_dir)
    output_dir  = Path(output_dir)

    sys.path.insert(0, str(Path(__file__).parent))
    from xview_converter import XVIEW_TO_ARGUS, convert_xview_to_yolo

    log.info("Step 1: Converting GeoJSON → YOLO labels...")
    raw_labels_dir = output_dir / "yolo_labels_raw"
    convert_xview_to_yolo(str(labels_path), str(images_dir), str(raw_labels_dir))

    log.info("Step 2: Stratified split...")
    by_image   = load_by_image(labels_path)
    disk_stems = {p.stem: p for p in images_dir.glob("*") if p.suffix in (".tif", ".tiff")}
    valid_ids  = [
        img_id for img_id in by_image
        if Path(img_id).stem in disk_stems
        and (raw_labels_dir / (Path(img_id).stem + ".txt")).exists()
    ]
    log.info(f"Valid images: {len(valid_ids)}")

    train_ids, val_ids = stratified_split(valid_ids, by_image, XVIEW_TO_ARGUS)
    log.info(f"Split: {len(train_ids)} train / {len(val_ids)} val")

    log.info("Step 3: Chipping images...")
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_chips = 0
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        out_img = output_dir / "images" / split
        out_lbl = output_dir / "labels" / split
        for i, img_id in enumerate(ids):
            img_path = disk_stems[Path(img_id).stem]
            lbl_path = raw_labels_dir / (Path(img_id).stem + ".txt")
            try:
                chips = chip_image(img_path, lbl_path, out_img, out_lbl)
                total_chips += chips
            except Exception as e:
                log.warning(f"Skipped {img_id}: {e}")
            if (i + 1) % 50 == 0:
                log.info(f"  {split}: {i+1}/{len(ids)} done")

    log.info(f"Total chips written: {total_chips:,}")

    log.info("Step 4: Writing dataset.yaml...")
    (output_dir / "dataset.yaml").write_text(
        f"path: {output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 5\n"
        "names: ['tank', 'large_vehicle', 'naval_vessel', 'aircraft', 'structure']\n"
    )
    log.info("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="path to xView_train.geojson")
    parser.add_argument("--images", required=True, help="directory with xView GeoTIFF images")
    parser.add_argument("--output", required=True, help="output directory for processed dataset")
    args = parser.parse_args()
    main(args.labels, args.images, args.output)
