"""
Audit Re-ID crop dataset quality without modifying source images.

Outputs:
  - CSV with per-image metrics and status: good / review / reject
  - Markdown summary by employee/class and reject reason

Usage:
  python tools/audit_reid_dataset_quality.py
  python tools/audit_reid_dataset_quality.py --root D:/warehouse_dataset/dataset_crops
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from config.settings import DATASET_CROPS_ROOT  # noqa: E402


DEFAULT_CLASSES = ["NV001", "NV002", "NV003", "NV004", "NV005"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dhash(gray: np.ndarray, hash_size: int = 8) -> str:
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = "".join("1" if v else "0" for v in diff.flatten())
    return f"{int(bits, 2):016x}"


def assess_image(
    image_path: Path,
    duplicate_seen: set[str],
    min_width: int,
    min_height: int,
    reject_blur: float,
    review_blur: float,
) -> dict:
    row = {
        "path": str(image_path),
        "file": image_path.name,
        "class": image_path.parent.name,
        "status": "reject",
        "reasons": "",
        "width": 0,
        "height": 0,
        "aspect_h_w": 0.0,
        "area": 0,
        "blur_laplacian": 0.0,
        "brightness": 0.0,
        "contrast": 0.0,
        "dhash": "",
    }

    img = cv2.imread(str(image_path))
    if img is None:
        row["reasons"] = "unreadable"
        return row

    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    aspect = float(height / width) if width else 0.0
    image_hash = dhash(gray)

    row.update(
        {
            "width": width,
            "height": height,
            "aspect_h_w": round(aspect, 4),
            "area": width * height,
            "blur_laplacian": round(blur, 4),
            "brightness": round(brightness, 4),
            "contrast": round(contrast, 4),
            "dhash": image_hash,
        }
    )

    reject_reasons = []
    review_reasons = []

    if width < min_width or height < min_height:
        reject_reasons.append("too_small")
    if aspect < 1.1 or aspect > 4.8:
        reject_reasons.append("bad_person_ratio")
    if blur < reject_blur:
        reject_reasons.append("too_blurry")
    elif blur < review_blur:
        review_reasons.append("soft_blur")
    if brightness < 25:
        reject_reasons.append("too_dark")
    elif brightness < 45:
        review_reasons.append("dark")
    if brightness > 235:
        reject_reasons.append("too_bright")
    elif brightness > 210:
        review_reasons.append("bright")
    if contrast < 15:
        reject_reasons.append("too_low_contrast")
    elif contrast < 25:
        review_reasons.append("low_contrast")
    if image_hash in duplicate_seen:
        review_reasons.append("possible_duplicate")
    else:
        duplicate_seen.add(image_hash)

    if reject_reasons:
        row["status"] = "reject"
        row["reasons"] = ";".join(reject_reasons + review_reasons)
    elif review_reasons:
        row["status"] = "review"
        row["reasons"] = ";".join(review_reasons)
    else:
        row["status"] = "good"
        row["reasons"] = ""

    return row


def iter_images(root: Path, classes: list[str]) -> list[Path]:
    paths = []
    for class_name in classes:
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                paths.append(path)
    return sorted(paths)


def write_summary(path: Path, rows: list[dict], root: Path, classes: list[str]) -> None:
    by_class = defaultdict(list)
    reason_counts = Counter()

    for row in rows:
        by_class[row["class"]].append(row)
        for reason in row["reasons"].split(";"):
            if reason:
                reason_counts[reason] += 1

    lines = [
        "# Re-ID Dataset Quality Audit",
        "",
        f"- root: `{root}`",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- classes: `{', '.join(classes)}`",
        f"- total_images: `{len(rows)}`",
        "",
        "## Status By Class",
        "",
        "| class | total | good | review | reject | keep_ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for class_name in classes:
        items = by_class.get(class_name, [])
        counts = Counter(row["status"] for row in items)
        total = len(items)
        keep = counts["good"] + counts["review"]
        keep_ratio = keep / total if total else 0.0
        lines.append(
            f"| {class_name} | {total} | {counts['good']} | {counts['review']} | "
            f"{counts['reject']} | {keep_ratio:.2%} |"
        )

    lines.extend(["", "## Top Reasons", "", "| reason | count |", "|---|---:|"])
    for reason, count in reason_counts.most_common():
        lines.append(f"| {reason} | {count} |")

    lines.extend(
        [
            "",
            "## Suggested Next Step",
            "",
            "Review rows with `status=review`, then train from images marked `good` plus manually approved review images.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DATASET_CROPS_ROOT)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--out-dir", type=Path, default=ROOT_DIR / "outputs" / "dataset_quality_audit")
    parser.add_argument("--min-width", type=int, default=32)
    parser.add_argument("--min-height", type=int, default=64)
    parser.add_argument("--reject-blur", type=float, default=20.0)
    parser.add_argument("--review-blur", type=float, default=65.0)
    args = parser.parse_args()

    root = args.root.expanduser()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    images = iter_images(root, args.classes)
    duplicate_seen_by_class: dict[str, set[str]] = defaultdict(set)
    rows = []

    for idx, image_path in enumerate(images, start=1):
        row = assess_image(
            image_path=image_path,
            duplicate_seen=duplicate_seen_by_class[image_path.parent.name],
            min_width=args.min_width,
            min_height=args.min_height,
            reject_blur=args.reject_blur,
            review_blur=args.review_blur,
        )
        rows.append(row)
        if idx % 500 == 0:
            print(f"[INFO] audited {idx}/{len(images)} images")

    csv_path = out_dir / "reid_dataset_quality.csv"
    md_path = out_dir / "reid_dataset_quality_summary.md"

    fieldnames = [
        "class",
        "status",
        "reasons",
        "width",
        "height",
        "aspect_h_w",
        "area",
        "blur_laplacian",
        "brightness",
        "contrast",
        "dhash",
        "file",
        "path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_summary(md_path, rows, root, args.classes)

    counts = Counter(row["status"] for row in rows)
    print("[DONE] Dataset audit complete")
    print("CSV    :", csv_path)
    print("Summary:", md_path)
    print("Status :", dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
