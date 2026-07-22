import csv
from pathlib import Path

import cv2
import numpy as np


BASE = Path(r"D:\warehouse_dataset_v2\face_train_layered")
AUDIT_DIR = Path(r"D:\warehouse_dataset_v2\audit_reports")

DETAIL_CSV = AUDIT_DIR / "face_pose_coverage_detail.csv"
READINESS_CSV = AUDIT_DIR / "face_pose_coverage_readiness.csv"

DST_LAYER = "L3_pose_variation"

# Chỉ flip để bù góc phải
ANGLE_PAIR = {
    "slight_right_img": "slight_left_img",
    "strong_right_img": "strong_left_img",
}


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img):
    ok, buf = cv2.imencode(".jpg", img)
    if ok:
        buf.tofile(str(path))
    return ok


def load_missing_targets():
    """
    Đọc readiness report để biết mỗi người còn thiếu bao nhiêu slight_right / strong_right.
    """
    targets = {}

    with open(READINESS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            person = row["person"]
            angle_type = row["angle_type"]
            missing = int(row["missing"])

            if angle_type not in ANGLE_PAIR:
                continue

            if missing <= 0:
                continue

            targets.setdefault(person, {})
            targets[person][angle_type] = missing

    return targets


def load_candidates():
    """
    Đọc detail report, gom ảnh theo person + yaw_bin.
    """
    candidates = {}

    with open(DETAIL_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            person = row["person"]
            yaw_bin = row["yaw_bin"]
            path = Path(row["path"])

            if yaw_bin not in ANGLE_PAIR.values():
                continue

            if not path.exists():
                continue

            candidates.setdefault(person, {})
            candidates[person].setdefault(yaw_bin, [])
            candidates[person][yaw_bin].append(path)

    return candidates


def main():
    if not DETAIL_CSV.exists():
        raise FileNotFoundError(f"Missing detail csv: {DETAIL_CSV}")

    if not READINESS_CSV.exists():
        raise FileNotFoundError(f"Missing readiness csv: {READINESS_CSV}")

    targets = load_missing_targets()
    candidates = load_candidates()

    print("=" * 90)
    print("FLIP MISSING RIGHT POSE")
    print(f"BASE: {BASE}")
    print(f"DETAIL: {DETAIL_CSV}")
    print(f"READINESS: {READINESS_CSV}")
    print("=" * 90)

    if not targets:
        print("No missing right-angle targets found. Nothing to do.")
        return

    for person, need_dict in targets.items():
        person_dir = BASE / person
        dst_dir = person_dir / DST_LAYER
        dst_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n===== {person} =====")

        for target_angle, missing_count in need_dict.items():
            source_angle = ANGLE_PAIR[target_angle]
            src_list = candidates.get(person, {}).get(source_angle, [])

            if not src_list:
                print(f"{target_angle}: no source images from {source_angle}")
                continue

            saved = 0

            for src_path in src_list:
                if saved >= missing_count:
                    break

                img = imread_unicode(src_path)
                if img is None:
                    continue

                flipped = cv2.flip(img, 1)

                out_name = (
                    f"FLIP_{source_angle}_TO_{target_angle}_"
                    f"{src_path.stem}.jpg"
                )
                out_path = dst_dir / out_name

                if out_path.exists():
                    continue

                if imwrite_unicode(out_path, flipped):
                    saved += 1

            print(
                f"{target_angle}: need {missing_count}, "
                f"source {source_angle}={len(src_list)}, saved {saved}"
            )

    print("\nDONE.")


if __name__ == "__main__":
    main()