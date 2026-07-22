import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import REID_MATCH_MARGIN, REID_MATCH_THRESHOLD  # noqa: E402
from reid.gallery import Gallery  # noqa: E402
from reid.reid_engine import ReIDEngine  # noqa: E402
from tools.test_synthetic_multi_person import format_top5, rank_reid  # noqa: E402


DEFAULT_ROOT = Path("D:/warehouse_dataset/synthetic_tests")
EMPLOYEE_IDS = ["NV001", "NV002", "NV003", "NV004", "NV005"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def load_unique_source_images(gt_dir: Path) -> list[dict]:
    seen = set()
    rows = []

    for gt_path in sorted(gt_dir.glob("*.json")):
        data = json.loads(gt_path.read_text(encoding="utf-8"))
        case_name = data.get("case_name", gt_path.stem)
        for frame in data.get("frames", []):
            for obj in frame.get("objects", []):
                emp_id = obj.get("emp_id", "")
                source_image = obj.get("source_image", "")
                key = (emp_id, source_image)
                if not emp_id or not source_image or key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "case_name": case_name,
                    "emp_id": emp_id,
                    "source_image": source_image,
                })

    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def write_markdown(path: Path, rows: list[dict]) -> None:
    valid = [r for r in rows if r["status"] == "ok"]
    correct = [r for r in valid if r["correct"] == "True"]
    unknown = [r for r in valid if r["pred_emp_id"] == "Unknown"]
    accuracy = len(correct) / len(valid) if valid else 0.0

    by_emp = defaultdict(list)
    for row in valid:
        by_emp[row["emp_id"]].append(row)

    confusions = Counter()
    for row in valid:
        pred = row["pred_emp_id"]
        if pred != row["emp_id"]:
            confusions[(row["emp_id"], pred)] += 1

    best_scores = [to_float(r["best_score"]) for r in valid]
    unknown_scores = [to_float(r["best_score"]) for r in unknown]
    unknown_margins = [to_float(r["margin"]) for r in unknown]

    lines = [
        "# Source Image Re-ID Sanity Check",
        "",
        "## Purpose",
        "",
        "Run Body Re-ID directly on `source_image` files from synthetic ground truth, without synthetic compositing, video decoding, YOLO crop, tracking, face, fusion, database, or camera.",
        "",
        "## Summary",
        "",
        f"- source_images_total: {len(rows)}",
        f"- readable_images: {len(valid)}",
        f"- correct: {len(correct)}",
        f"- accuracy: {accuracy:.4f}",
        f"- unknown_count: {len(unknown)}",
        f"- REID_MATCH_THRESHOLD: {REID_MATCH_THRESHOLD:.6f}",
        f"- REID_MATCH_MARGIN: {REID_MATCH_MARGIN:.6f}",
        f"- best_score_mean: {float(np.mean(best_scores)):.6f}" if best_scores else "- best_score_mean: 0.000000",
        f"- unknown_best_score_mean: {float(np.mean(unknown_scores)):.6f}" if unknown_scores else "- unknown_best_score_mean: 0.000000",
        f"- unknown_margin_mean: {float(np.mean(unknown_margins)):.6f}" if unknown_margins else "- unknown_margin_mean: 0.000000",
        "",
        "## Per Employee",
        "",
        "| emp_id | total | correct | accuracy | unknown |",
        "|---|---:|---:|---:|---:|",
    ]

    for emp_id in EMPLOYEE_IDS:
        emp_rows = by_emp.get(emp_id, [])
        emp_correct = [r for r in emp_rows if r["correct"] == "True"]
        emp_unknown = [r for r in emp_rows if r["pred_emp_id"] == "Unknown"]
        emp_acc = len(emp_correct) / len(emp_rows) if emp_rows else 0.0
        lines.append(f"| {emp_id} | {len(emp_rows)} | {len(emp_correct)} | {emp_acc:.4f} | {len(emp_unknown)} |")

    lines.extend(["", "## Common Confusions", ""])
    if confusions:
        for (gt, pred), count in confusions.most_common(10):
            lines.append(f"- {gt}->{pred}: {count}")
    else:
        lines.append("- None")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- If this sanity accuracy is high but synthetic test accuracy is low, the next suspect is synthetic cutout/scale/background or YOLO crop quality.",
        "- If this sanity accuracy is also low, the next suspect is gallery/model/mapping/threshold/API compatibility.",
    ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    gt_dir = args.synthetic_root / "ground_truth"
    reports_dir = args.synthetic_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading Body Re-ID engine and gallery")
    reid = ReIDEngine()
    gallery = Gallery()
    gallery_data = gallery.all()
    if not gallery_data:
        raise RuntimeError("Gallery is empty. Expected python_cv/data/gallery/gallery.pkl")
    print(f"[INFO] Gallery employees: {sorted(gallery_data.keys())}")

    source_rows = load_unique_source_images(gt_dir)
    rows = []

    for item in source_rows:
        image_path = Path(item["source_image"])
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            rows.append({
                "case_name": item["case_name"],
                "emp_id": item["emp_id"],
                "source_image": str(image_path),
                "status": "unreadable",
                "image_shape": "",
                "pred_emp_id": "Unknown",
                "correct": "False",
                "best_emp_id": "",
                "best_score": "0.000000",
                "second_emp_id": "",
                "second_score": "0.000000",
                "margin": "0.000000",
                "top5_scores": "",
                "reid_threshold_used": f"{REID_MATCH_THRESHOLD:.6f}",
                "reid_margin_used": f"{REID_MATCH_MARGIN:.6f}",
                "reject_reason": "invalid_crop",
                "reid_call_method": "get_embedding+match_score",
            })
            continue

        result = rank_reid(reid, gallery_data, frame)
        pred = result["pred_emp_id"] or "Unknown"
        rows.append({
            "case_name": item["case_name"],
            "emp_id": item["emp_id"],
            "source_image": str(image_path),
            "status": "ok",
            "image_shape": f"{frame.shape[1]}x{frame.shape[0]}",
            "pred_emp_id": pred,
            "correct": str(pred == item["emp_id"]),
            "best_emp_id": result["best_emp_id"] or "",
            "best_score": f"{result['best_score']:.6f}",
            "second_emp_id": result["second_emp_id"] or "",
            "second_score": f"{result['second_score']:.6f}",
            "margin": f"{result['margin']:.6f}",
            "top5_scores": format_top5(result["top5_scores"]),
            "reid_threshold_used": f"{REID_MATCH_THRESHOLD:.6f}",
            "reid_margin_used": f"{REID_MATCH_MARGIN:.6f}",
            "reject_reason": result["reject_reason"],
            "reid_call_method": result["reid_call_method"],
        })

    fields = [
        "case_name",
        "emp_id",
        "source_image",
        "status",
        "image_shape",
        "pred_emp_id",
        "correct",
        "best_emp_id",
        "best_score",
        "second_emp_id",
        "second_score",
        "margin",
        "top5_scores",
        "reid_threshold_used",
        "reid_margin_used",
        "reject_reason",
        "reid_call_method",
    ]
    write_csv(reports_dir / "source_image_reid_sanity.csv", rows, fields)
    write_markdown(reports_dir / "source_image_reid_sanity.md", rows)

    valid = [r for r in rows if r["status"] == "ok"]
    correct = [r for r in valid if r["correct"] == "True"]
    accuracy = len(correct) / len(valid) if valid else 0.0
    print(f"[DONE] source images={len(rows)} readable={len(valid)} accuracy={accuracy:.3f}")
    print(f"[DONE] Reports: {reports_dir}")


if __name__ == "__main__":
    main()
