"""
ARGUS TARGETING SYSTEM - Sentinel-2 Ingestion Pipeline
"""

import os
import math
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("argus.ingest")


@dataclass
class IngestionConfig:
    client_id: str = ""
    client_secret: str = ""
    bbox_coords: tuple = (35.85, 34.85, 35.95, 34.95)
    time_start: str = "2025-01-01"
    time_end: str = "2025-06-01"
    resolution: float = 10.0
    max_cloud_cover: float = 15.0
    chip_size: int = 640
    chip_overlap: int = 64
    upscale_factor: int = 4
    output_dir: Path = Path("data/sentinel_tiles")
    chip_dir: Path = Path("data/chips")

    def __post_init__(self):
        if not self.client_id:
            self.client_id = os.getenv("SENTINEL_HUB_CLIENT_ID", "")
        if not self.client_secret:
            self.client_secret = os.getenv("SENTINEL_HUB_CLIENT_SECRET", "")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chip_dir.mkdir(parents=True, exist_ok=True)


class SentinelIngestionPipeline:
    def __init__(self, config: IngestionConfig):
        self.config = config

    @staticmethod
    def tile_image(image: np.ndarray, chip_size: int = 640, overlap: int = 64) -> list[dict]:
        h, w = image.shape[:2]
        stride = chip_size - overlap
        chips = []
        n_rows = max(1, math.ceil((h - overlap) / stride))
        n_cols = max(1, math.ceil((w - overlap) / stride))

        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                y_off = max(0, min(row_idx * stride, h - chip_size))
                x_off = max(0, min(col_idx * stride, w - chip_size))
                y_end = min(y_off + chip_size, h)
                x_end = min(x_off + chip_size, w)
                chip = image[y_off:y_end, x_off:x_end]

                if chip.shape[0] < chip_size or chip.shape[1] < chip_size:
                    padded = np.zeros((chip_size, chip_size, image.shape[2]), dtype=image.dtype)
                    padded[:chip.shape[0], :chip.shape[1]] = chip
                    chip = padded

                chips.append({"chip": chip, "row": row_idx, "col": col_idx,
                               "x_off": x_off, "y_off": y_off})

        return chips

    @staticmethod
    def upscale_chip(chip: np.ndarray, factor: int = 4) -> np.ndarray:
        img = Image.fromarray(chip)
        new_size = (img.width * factor, img.height * factor)
        return np.array(img.resize(new_size, Image.NEAREST))


def chip_pixel_to_wgs84(pixel_x, pixel_y, chip_x_off, chip_y_off,
                         bbox, tile_width, tile_height, upscale_factor=1):
    src_x = chip_x_off + (pixel_x / upscale_factor)
    src_y = chip_y_off + (pixel_y / upscale_factor)
    min_lon, min_lat, max_lon, max_lat = bbox
    lon = min_lon + (src_x / tile_width) * (max_lon - min_lon)
    lat = max_lat - (src_y / tile_height) * (max_lat - min_lat)
    return (lon, lat)
