import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np


PEOPLE = ["NV001_Men", "NV002_Duc"]

LAYERS = [
    "L1_easy_front_clear",
    "L2_normal_clean",
    "L3_pose_variation",
    "L4_hard_valid",
    "L5_extreme_hard_valid",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_original_scores(filename: str):
    """
    Parse score gốc từ tên file kiểu:
    ..._q0.626_s0.866.jpg
    """
    m = re.search(r"_q([0-9.]+)_s([0-9.]+)\.(jpg|jpeg|png|bmp|webp)$", filename, re.IGNORECASE)
    if not m:
        return None, None

    try:
        return float(m.group(1)), float(m.group(2))
    except Exception:
        return None, None


def laplacian_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_contrast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)), float(np.std(gray))


def image_hash_quick(img):
    """
    Hash đơn giản để phát hiện ảnh trùng/na ná.
    Không phải perceptual hash hoàn hảo, nhưng đủ rà nhanh.
    """
    small = cv2.resize(img, (16, 16))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mean = gray.mean()
    bits = (gray > mean).astype(np.uint8).flatten()
    return "".join(map(str, bits.tolist()))


def collect_images(root: Path):
    files = []
    for person in PEOPLE:
        for layer in LAYERS:
            layer_dir = root / person / layer
            if not layer_dir.exists():
                continue

            for p in layer_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    files.append((person, layer, p))

    return files


def expected_layer_weight(layer):
    if layer.startswith("L1"):
        return 1.0
    if layer.startswith("L2"):
        return 1.2
    if layer.startswith("L3"):
        return 1.5
    if layer.startswith("L4"):
        return 2.5
    if layer.startswith("L5"):
        return 3.5
    return 1.0


def quality_flag(layer, sharpness, brightness, contrast, orig_s):
    flags = []

    if sharpness < 8:
        flags.append("TOO_BLUR")
    elif sharpness < 18:
        flags.append("LOW_SHARPNESS")

    if brightness < 45:
        flags.append("TOO_DARK")
    elif brightness > 225:
        flags.append("TOO_BRIGHT")

    if contrast < 15:
        flags.append("LOW_CONTRAST")

    if orig_s is not None and orig_s < 0.50:
        flags.append("LOW_ORIG_FACE_SCORE")

    # Layer-specific warning
    if layer.startswith("L1"):
        if sharpness < 35:
            flags.append("L1_SHARPNESS_WEAK")
        if orig_s is not None and orig_s < 0.75:
            flags.append("L1_ORIG_SCORE_WEAK")

    if layer.startswith("L2"):
        if sharpness < 25:
            flags.append("L2_SHARPNESS_WEAK")
        if orig_s is not None and orig_s < 0.65:
            flags.append("L2_ORIG_SCORE_WEAK")

    if layer.startswith("L3"):
        if sharpness < 15:
            flags.append("L3_TOO_WEAK")

    if layer.startswith("L4"):
        # L4 hard valid: khó nhưng vẫn không được nát.
        if sharpness < 10:
            flags.append("L4_TOO_BLUR_FOR_TRAIN")

    if layer.startswith("L5"):
        # L5 hard valid: cực khó nhưng vẫn phải có tín hiệu.
        if sharpness < 8:
            flags.append("L5_REJECT_CANDIDATE")

    return "|".join(flags) if flags else "OK"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=r"D:\warehouse_dataset_v2\face_train_layered",
        help="Dataset root",
    )
    parser.add_argument(
        "--out-dir",
        default=r"D:\warehouse_dataset_v2\audit_reports",
        help="Output report directory",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(root)

    if not images:
        print(f"[ERROR] No images found in {root}")
        return

    detail_csv = out_dir / "face_dataset_quality_detail.csv"
    summary_csv = out_dir / "face_dataset_quality_summary.csv"
    duplicate_csv = out_dir / "face_dataset_possible_duplicates.csv"

    summary = {}
    hashes = {}

    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "person",
            "layer",
            "filename",
            "path",
            "orig_q",
            "orig_s",
            "sharpness",
            "brightness",
            "contrast",
            "img_w",
            "img_h",
            "layer_weight",
            "flags",
        ])

        for person, layer, path in images:
            img = cv2.imread(str(path))
            if img is None:
                row = [person, layer, path.name, str(path), "", "", "", "", "", "", "", expected_layer_weight(layer), "READ_FAIL"]
                writer.writerow(row)
                continue

            h, w = img.shape[:2]
            sharp = laplacian_sharpness(img)
            bright, contrast = brightness_contrast(img)
            orig_q, orig_s = parse_original_scores(path.name)
            flags = quality_flag(layer, sharp, bright, contrast, orig_s)
            weight = expected_layer_weight(layer)

            writer.writerow([
                person,
                layer,
                path.name,
                str(path),
                "" if orig_q is None else f"{orig_q:.4f}",
                "" if orig_s is None else f"{orig_s:.4f}",
                f"{sharp:.2f}",
                f"{bright:.2f}",
                f"{contrast:.2f}",
                w,
                h,
                weight,
                flags,
            ])

            key = (person, layer)
            if key not in summary:
                summary[key] = {
                    "count": 0,
                    "sharp": [],
                    "bright": [],
                    "contrast": [],
                    "ok": 0,
                    "warn": 0,
                }

            summary[key]["count"] += 1
            summary[key]["sharp"].append(sharp)
            summary[key]["bright"].append(bright)
            summary[key]["contrast"].append(contrast)

            if flags == "OK":
                summary[key]["ok"] += 1
            else:
                summary[key]["warn"] += 1

            ih = image_hash_quick(img)
            hashes.setdefault(ih, []).append(str(path))

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "person",
            "layer",
            "count",
            "ok_count",
            "warning_count",
            "sharp_mean",
            "sharp_min",
            "sharp_max",
            "brightness_mean",
            "contrast_mean",
            "layer_weight",
        ])

        for person in PEOPLE:
            for layer in LAYERS:
                key = (person, layer)
                data = summary.get(key)

                if not data:
                    writer.writerow([person, layer, 0, 0, 0, "", "", "", "", "", expected_layer_weight(layer)])
                    continue

                sharp_arr = np.array(data["sharp"], dtype=float)
                bright_arr = np.array(data["bright"], dtype=float)
                contrast_arr = np.array(data["contrast"], dtype=float)

                writer.writerow([
                    person,
                    layer,
                    data["count"],
                    data["ok"],
                    data["warn"],
                    f"{sharp_arr.mean():.2f}",
                    f"{sharp_arr.min():.2f}",
                    f"{sharp_arr.max():.2f}",
                    f"{bright_arr.mean():.2f}",
                    f"{contrast_arr.mean():.2f}",
                    expected_layer_weight(layer),
                ])

    with open(duplicate_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["hash", "count", "paths"])

        dup_count = 0
        for ih, paths in hashes.items():
            if len(paths) >= 2:
                dup_count += 1
                writer.writerow([ih, len(paths), " | ".join(paths)])

    print("=" * 90)
    print("FACE DATASET AUDIT DONE")
    print(f"Root: {root}")
    print(f"Total images: {len(images)}")
    print(f"Detail report: {detail_csv}")
    print(f"Summary report: {summary_csv}")
    print(f"Possible duplicates: {duplicate_csv}")
    print("=" * 90)

    print("\nSUMMARY:")
    for person in PEOPLE:
        print(f"\n===== {person} =====")
        total = 0
        for layer in LAYERS:
            key = (person, layer)
            count = summary.get(key, {}).get("count", 0)
            total += count
            print(f"{layer}: {count}")
        print(f"TOTAL: {total}")


if __name__ == "__main__":
    main()