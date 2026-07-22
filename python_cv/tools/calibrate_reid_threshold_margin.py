import argparse
import csv
import random
import sys
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from reid.gallery import Gallery
from reid.reid_engine import ReIDEngine
from config.settings import DATASET_CROPS_ROOT


DEFAULT_DATASET_ROOT = DATASET_CROPS_ROOT
EMPLOYEE_IDS = ["NV001", "NV002", "NV003", "NV004", "NV005"]
CONFIGS = [
    (0.90, 0.05),
    (0.90, 0.03),
    (0.90, 0.02),
    (0.90, 0.015),
    (0.93, 0.02),
    (0.93, 0.015),
    (0.95, 0.015),
]


def list_images(dataset_root: Path, employee_id: str, max_per_class: int) -> list[Path]:
    folder = dataset_root / employee_id
    if not folder.exists():
        print(f"[WARN] Missing folder: {folder}")
        return []

    images = sorted(folder.glob("*.jpg"))

    if employee_id != "NV005":
        return random.sample(images, min(max_per_class, len(images)))

    city = [p for p in images if p.name.startswith("NV005_city")]
    non_city = [p for p in images if not p.name.startswith("NV005_city")]

    if len(city) >= max_per_class:
        return random.sample(city, max_per_class)

    selected = list(city)
    remaining = max_per_class - len(selected)
    selected.extend(random.sample(non_city, min(remaining, len(non_city))))
    random.shuffle(selected)
    return selected


def rank_image(reid: ReIDEngine, gallery_data: dict, image_path: Path) -> dict | None:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[WARN] Cannot read image: {image_path}")
        return None

    query_emb = reid.get_embedding(image)
    if query_emb is None:
        print(f"[WARN] Cannot extract embedding: {image_path}")
        return None

    rows = []
    for employee_id, embeds in gallery_data.items():
        score = reid.match_score(query_emb, embeds)
        rows.append((employee_id, score))

    if not rows:
        return None

    rows.sort(key=lambda x: x[1], reverse=True)

    best_id, best_score = rows[0]
    second_id, second_score = rows[1] if len(rows) > 1 else ("", 0.0)

    return {
        "image_path": str(image_path),
        "best_id": best_id,
        "best_score": float(best_score),
        "second_id": second_id,
        "second_score": float(second_score),
        "margin": float(best_score - second_score),
    }


def summarize(results: list[dict], threshold: float, margin_threshold: float) -> dict:
    total = len(results)
    top1_correct = sum(1 for r in results if r["best_id"] == r["true_id"])

    confirmed = [
        r for r in results
        if r["best_score"] >= threshold and r["margin"] >= margin_threshold
    ]
    correct_confirm = sum(1 for r in confirmed if r["best_id"] == r["true_id"])
    wrong_confirm = len(confirmed) - correct_confirm
    unknown = total - len(confirmed)

    nv005_rows = [r for r in results if r["true_id"] == "NV005"]
    nv005_total = len(nv005_rows)
    nv005_top1 = sum(1 for r in nv005_rows if r["best_id"] == "NV005")
    nv005_confirmed = [
        r for r in nv005_rows
        if r["best_score"] >= threshold and r["margin"] >= margin_threshold
    ]
    nv005_correct_confirm = sum(1 for r in nv005_confirmed if r["best_id"] == "NV005")
    nv005_wrong_confirm = len(nv005_confirmed) - nv005_correct_confirm
    nv005_unknown = nv005_total - len(nv005_confirmed)

    return {
        "threshold": threshold,
        "margin_threshold": margin_threshold,
        "total_images": total,
        "top1_accuracy": top1_correct / total if total else 0.0,
        "confirm_rate": len(confirmed) / total if total else 0.0,
        "correct_confirm_count": correct_confirm,
        "wrong_confirm_count": wrong_confirm,
        "unknown_count": unknown,
        "nv005_total_images": nv005_total,
        "nv005_top1_accuracy": nv005_top1 / nv005_total if nv005_total else 0.0,
        "nv005_confirm_rate": len(nv005_confirmed) / nv005_total if nv005_total else 0.0,
        "nv005_wrong_confirm_count": nv005_wrong_confirm,
        "nv005_unknown_count": nv005_unknown,
    }


def write_results_csv(results: list[dict], path: Path) -> None:
    fieldnames = [
        "true_id",
        "image_path",
        "best_id",
        "best_score",
        "second_id",
        "second_score",
        "margin",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in fieldnames})


def write_summary_csv(summary_rows: list[dict], path: Path) -> None:
    fieldnames = [
        "threshold",
        "margin_threshold",
        "total_images",
        "top1_accuracy",
        "confirm_rate",
        "correct_confirm_count",
        "wrong_confirm_count",
        "unknown_count",
        "nv005_total_images",
        "nv005_top1_accuracy",
        "nv005_confirm_rate",
        "nv005_wrong_confirm_count",
        "nv005_unknown_count",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def print_summary(summary_rows: list[dict]) -> None:
    print()
    print("=" * 100)
    print("RE-ID THRESHOLD / MARGIN CALIBRATION SUMMARY")
    print("=" * 100)
    header = (
        "threshold margin total top1_acc confirm_rate correct wrong unknown "
        "nv005_top1 nv005_confirm nv005_wrong nv005_unknown"
    )
    print(header)
    print("-" * 100)

    for row in summary_rows:
        print(
            f"{row['threshold']:.3f} "
            f"{row['margin_threshold']:.3f} "
            f"{row['total_images']:5d} "
            f"{row['top1_accuracy']:.3f} "
            f"{row['confirm_rate']:.3f} "
            f"{row['correct_confirm_count']:7d} "
            f"{row['wrong_confirm_count']:5d} "
            f"{row['unknown_count']:7d} "
            f"{row['nv005_top1_accuracy']:.3f} "
            f"{row['nv005_confirm_rate']:.3f} "
            f"{row['nv005_wrong_confirm_count']:10d} "
            f"{row['nv005_unknown_count']:12d}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate body Re-ID threshold/margin using direct crops."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset crop root containing NV001..NV005 folders.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=50,
        help="Maximum random images per employee class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "outputs" / "reid_calibration",
        help="Output directory for calibration CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "reid_calibration_results.csv"
    summary_path = args.output_dir / "reid_calibration_summary.csv"

    print("=" * 100)
    print("BODY RE-ID DIRECT CROP CALIBRATION")
    print("=" * 100)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Max per class: {args.max_per_class}")
    print(f"Seed: {args.seed}")
    print(f"Output dir: {args.output_dir}")

    image_items = []
    for employee_id in EMPLOYEE_IDS:
        paths = list_images(args.dataset_root, employee_id, args.max_per_class)
        print(f"{employee_id}: selected {len(paths)} images")
        image_items.extend((employee_id, p) for p in paths)

    reid = ReIDEngine()
    gallery = Gallery()
    gallery_data = gallery.all()

    results = []
    total = len(image_items)

    for idx, (true_id, image_path) in enumerate(image_items, start=1):
        ranked = rank_image(reid, gallery_data, image_path)
        if ranked is None:
            continue

        ranked["true_id"] = true_id
        results.append(ranked)

        print(
            f"[{idx:03d}/{total:03d}] {true_id} "
            f"best={ranked['best_id']} score={ranked['best_score']:.4f} "
            f"second={ranked['second_id']} second_score={ranked['second_score']:.4f} "
            f"margin={ranked['margin']:.4f}"
        )

    summary_rows = [summarize(results, threshold, margin) for threshold, margin in CONFIGS]

    write_results_csv(results, results_path)
    write_summary_csv(summary_rows, summary_path)
    print_summary(summary_rows)

    print()
    print(f"Saved results: {results_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
