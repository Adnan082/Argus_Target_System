"""
Local smoke test — validates the preprocessing pipeline on a tiny synthetic sample.
Run this BEFORE pushing to EC2.

Usage:
    python scripts/test_local.py

What it tests:
  1. xview_converter.py: synthetic GeoJSON + dummy images → YOLO label files
  2. Label format: values in [0,1], correct column count
  3. train.py can be imported without crashing
"""

import json
import sys
import tempfile
import logging
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("smoke_test")

PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        log.info(f"  PASS  {name}")
    else:
        FAIL.append(name)
        log.error(f"  FAIL  {name}  {detail}")


# ---------------------------------------------------------------------------
# Test 1: converter produces valid YOLO labels from synthetic data
# ---------------------------------------------------------------------------
def test_converter():
    log.info("--- Test: xview_converter ---")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.preprocessing.xview_converter import convert_xview_to_yolo

    with tempfile.TemporaryDirectory() as tmp:
        img_dir = Path(tmp) / "images"
        lbl_out = Path(tmp) / "labels"
        img_dir.mkdir()

        # Create two tiny dummy images
        for name in ["001.tif", "002.tif"]:
            img = Image.new("RGB", (200, 200), color=(100, 100, 100))
            img.save(img_dir / name)

        # Synthetic GeoJSON with known xView type_ids
        geojson = {
            "features": [
                {
                    "properties": {"image_id": "001.tif", "type_id": 71},  # tank
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[10, 10], [40, 10], [40, 40], [10, 40], [10, 10]]],
                    },
                },
                {
                    "properties": {"image_id": "001.tif", "type_id": 23},  # large_vehicle
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[100, 100], [150, 100], [150, 150], [100, 150], [100, 100]]],
                    },
                },
                {
                    "properties": {"image_id": "002.tif", "type_id": 999},  # unknown — should be skipped
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[5, 5], [20, 5], [20, 20], [5, 20], [5, 5]]],
                    },
                },
                {
                    "properties": {"image_id": "missing.tif", "type_id": 71},  # no image file
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[5, 5], [20, 5], [20, 20], [5, 20], [5, 5]]],
                    },
                },
            ]
        }
        labels_file = Path(tmp) / "labels.geojson"
        labels_file.write_text(json.dumps(geojson))

        n = convert_xview_to_yolo(str(labels_file), str(img_dir), str(lbl_out))
        check("converter returns count > 0", n > 0, f"got {n}")

        label_001 = lbl_out / "001.txt"
        check("001.txt created", label_001.exists())

        if label_001.exists():
            lines = label_001.read_text().strip().splitlines()
            check("001.txt has 2 boxes", len(lines) == 2, f"got {len(lines)}")

            for line in lines:
                parts = line.split()
                check("label has 5 columns", len(parts) == 5, line)
                cls, cx, cy, w, h = int(parts[0]), *[float(p) for p in parts[1:]]
                check(f"cls in [0,4]", 0 <= cls <= 4, f"cls={cls}")
                for val, name in zip([cx, cy, w, h], ["cx", "cy", "w", "h"]):
                    check(f"{name} in [0,1]", 0.0 <= val <= 1.0, f"{name}={val}")

        label_002 = lbl_out / "002.txt"
        check("002.txt not created (all boxes unknown class)", not label_002.exists())


# ---------------------------------------------------------------------------
# Test 2: train.py imports cleanly
# ---------------------------------------------------------------------------
def test_train_import():
    log.info("--- Test: train.py import ---")
    try:
        from src.training import train  # noqa: F401
        check("train.py imports without error", True)
    except ImportError as e:
        # ultralytics not installed locally → acceptable
        if "ultralytics" in str(e):
            log.info("  SKIP  train import (ultralytics not installed — expected locally)")
        else:
            check("train.py imports without error", False, str(e))
    except Exception as e:
        check("train.py imports without error", False, str(e))


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_converter()
    test_train_import()

    print()
    print(f"Results: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)
    else:
        print("All checks passed — safe to push to EC2.")
