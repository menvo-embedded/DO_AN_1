import argparse
import csv
import sys
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from reid.face_insightface_engine import InsightFaceEngine


def latest_zone2_crop_dir() -> Path:
    debug_root = ROOT_DIR / "outputs" / "zone2_reid_debug"
    runs = [p for p in debug_root.glob("*") if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No Zone2 debug runs found in {debug_root}")

    latest = max(runs, key=lambda p: p.stat().st_mtime)
    crops = latest / "crops"
    if not crops.exists():
        raise FileNotFoundError(f"Missing crops folder: {crops}")
    return crops


def decision_from_result(result: dict) -> str:
    status = result.get("status", "")
    if status == "MATCH":
        return f"MATCH:{result.get('employee_id')}"
    return status or "UNKNOWN"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run InsightFace directly on saved Zone2 person crops."
    )
    parser.add_argument(
        "crop_dir",
        nargs="?",
        type=Path,
        default=None,
        help="Crop directory. Defaults to latest outputs/zone2_reid_debug/<run>/crops.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max crops to test. 0 = all.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="CSV output path. Defaults to crop_dir parent/zone2_insightface_crop_results.csv.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    crop_dir = args.crop_dir or latest_zone2_crop_dir()
    output_csv = args.output_csv or crop_dir.parent / "zone2_insightface_crop_results.csv"

    images = sorted(crop_dir.glob("*.jpg"))
    if args.limit > 0:
        images = images[: args.limit]

    print("=" * 100)
    print("ZONE2 SAVED CROP INSIGHTFACE TEST")
    print("=" * 100)
    print("Crop dir:", crop_dir)
    print("Images:", len(images))
    print("CSV:", output_csv)

    engine = InsightFaceEngine(
        gallery_path=str(ROOT_DIR / "data" / "face_gallery" / "insightface_gallery.pkl"),
        model_name="buffalo_sc",
        use_gpu=True,
    )

    fieldnames = [
        "crop_path",
        "face_detected",
        "best_face_id",
        "face_score",
        "second_score",
        "margin",
        "det_score",
        "bbox",
        "decision",
        "ranking",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for crop_path in images:
            image = cv2.imread(str(crop_path))
            if image is None:
                print("Cannot read:", crop_path)
                continue

            result = engine.identify_image(image)
            face_detected = result.get("status") != "NO_FACE"
            ranking = result.get("ranking", [])
            best = ranking[0] if ranking else {}
            best_face_id = best.get("employee_id", result.get("employee_id", "Unknown"))
            face_score = float(result.get("score", 0.0))
            second_score = float(result.get("second_score", -1.0))
            margin = float(result.get("margin", 0.0))
            det_score = result.get("det_score", "")
            bbox = result.get("bbox")
            decision = decision_from_result(result)
            ranking_text = ";".join(
                f"{r.get('employee_id')}:{float(r.get('score', 0.0)):.6f}"
                for r in ranking
            )

            row = {
                "crop_path": str(crop_path),
                "face_detected": face_detected,
                "best_face_id": best_face_id,
                "face_score": f"{face_score:.6f}",
                "second_score": f"{second_score:.6f}",
                "margin": f"{margin:.6f}",
                "det_score": f"{float(det_score):.6f}" if det_score != "" else "",
                "bbox": bbox,
                "decision": decision,
                "ranking": ranking_text,
            }
            writer.writerow(row)

            print(
                f"{crop_path.name} | face_detected={face_detected} "
                f"best_face_id={best_face_id} score={face_score:.4f} "
                f"second={second_score:.4f} margin={margin:.4f} "
                f"decision={decision}"
            )

    print()
    print("Saved CSV:", output_csv)


if __name__ == "__main__":
    main()
