#!/usr/bin/env python3
"""
Create a Qualcomm QNN calibration dataset for YOLO26n-cls.

Steps:
  1. Read images directly from dataset_img directory.
  2. Preprocess with the same ULTRALYTICS flow used by
     imagenette320_512/build_imagenette320_512_nhwc_raw.py and save NHWC
     float32 .raw tensors to dataset_raw.
  3. Write absolute raw paths to datasets.txt for QNN conversion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


IMGSZ = 640
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


def collect_images_from_dir(img_dir: Path) -> list[Path]:
    """Collect all image files from a flat directory, sorted."""
    if not img_dir.is_dir():
        raise FileNotFoundError(f"image directory not found: {img_dir}")

    images = sorted(
        p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXT
    )
    if not images:
        raise RuntimeError(f"no images found in: {img_dir}")

    print(f"Found {len(images)} images in {img_dir}")
    return images


def preprocess_ultralytics(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot read image: {path}")
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32) / 255.0
    return np.ascontiguousarray(x)  # HWC float32, one sample is NHWC without batch.


def write_raw_dataset(
    images: list[Path],
    dataset_raw_dir: Path,
    dataset_list: Path,
) -> int:
    dataset_raw_dir.mkdir(parents=True, exist_ok=True)

    raw_paths: list[Path] = []
    for idx, img in enumerate(images):
        try:
            tensor = preprocess_ultralytics(img)
        except ValueError as exc:
            print(f"[skip] {exc}", file=sys.stderr)
            continue

        raw_path = dataset_raw_dir / f"calib_{idx:05d}_{img.stem}.raw"
        tensor.tofile(raw_path)
        raw_paths.append(raw_path.resolve())

    dataset_list.parent.mkdir(parents=True, exist_ok=True)
    dataset_list.write_text(
        "\n".join(str(p) for p in raw_paths) + ("\n" if raw_paths else ""),
        encoding="utf-8",
    )

    return len(raw_paths)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Create QNN calibration raw tensors and datasets.txt from existing images"
    )
    parser.add_argument(
        "--dataset-img-dir",
        type=Path,
        default=here / "dataset_img",
        help="Directory containing calibration images to process",
    )
    parser.add_argument(
        "--dataset-raw-dir",
        type=Path,
        default=here / "dataset_raw",
        help="Directory to write preprocessed float32 NHWC raw files into",
    )
    parser.add_argument(
        "--dataset-list",
        type=Path,
        default=here / "datasets.txt",
        help="Output QNN input list containing absolute .raw paths",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        images = collect_images_from_dir(args.dataset_img_dir.resolve())
        raw_count = write_raw_dataset(
            images,
            args.dataset_raw_dir.resolve(),
            args.dataset_list.resolve(),
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    elems = IMGSZ * IMGSZ * 3
    bytes_per = elems * np.dtype(np.float32).itemsize
    print(f"Processed images: {len(images)} <- {args.dataset_img_dir.resolve()}")
    print(f"Raw files: {raw_count} -> {args.dataset_raw_dir.resolve()}")
    print(f"Dataset list: {args.dataset_list.resolve()}")
    print(f"Per raw: {IMGSZ}x{IMGSZ}x3 NHWC float32, {elems} elems, {bytes_per} bytes")


if __name__ == "__main__":
    main()
