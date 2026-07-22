import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


PEOPLE = ["NV001_Men", "NV002_Duc"]

LAYERS = [
    "L1_easy_front_clear",
    "L2_normal_clean",
    "L3_pose_variation",
    "L4_hard_valid",
    "L5_extreme_hard_valid",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def init_face_app(model_name, det_size, det_thresh):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    print("=" * 90)
    print("Loading InsightFace pose audit")
    print(f"model_name={model_name}")
    print(f"det_size={det_size}")
    print(f"det_thresh={det_thresh}")
    print(f"providers={providers}")
    print("=" * 90)

    app = FaceAnalysis(
        name=model_name,
        providers=providers,
        allowed_modules=["detection"],
    )

    app.prepare(
        ctx_id=0,
        det_size=det_size,
        det_thresh=det_thresh,
    )

    return app


def largest_face(faces):
    if not faces:
        return None

    return sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )[0]


def calc_pose_proxy(face):
    """
    Dùng 5 landmarks của InsightFace:
    0 left_eye, 1 right_eye, 2 nose, 3 left_mouth, 4 right_mouth

    Đây là proxy, không phải pose 3D tuyệt đối.
    Nhưng đủ để audit dataset có đủ góc mặt hay chưa.
    """
    if not hasattr(face, "kps") or face.kps is None or len(face.kps) < 5:
        return None

    kps = face.kps.astype(float)

    left_eye = kps[0]
    right_eye = kps[1]
    nose = kps[2]
    left_mouth = kps[3]
    right_mouth = kps[4]

    eye_center = (left_eye + right_eye) / 2.0
    mouth_center = (left_mouth + right_mouth) / 2.0

    eye_dist = float(np.linalg.norm(right_eye - left_eye)) + 1e-6

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    roll_deg = math.degrees(math.atan2(dy, dx))
    roll_abs = abs(roll_deg)

    # yaw proxy: mũi lệch khỏi tâm hai mắt
    yaw_signed = float(nose[0] - eye_center[0]) / eye_dist
    yaw_abs = abs(yaw_signed)

    # pitch proxy: vị trí mũi trong đoạn mắt -> miệng
    vertical = float(mouth_center[1] - eye_center[1])
    if abs(vertical) < 1e-6:
        pitch_proxy = 0.5
    else:
        pitch_proxy = float(nose[1] - eye_center[1]) / vertical

    return {
        "yaw_signed": yaw_signed,
        "yaw_abs": yaw_abs,
        "roll_deg": roll_deg,
        "roll_abs": roll_abs,
        "pitch_proxy": pitch_proxy,
    }


def classify_yaw(yaw_signed, yaw_abs):
    if yaw_abs <= 0.25:
        return "front"

    if yaw_abs <= 0.55:
        return "slight_right_img" if yaw_signed > 0 else "slight_left_img"

    return "strong_right_img" if yaw_signed > 0 else "strong_left_img"


def classify_pitch(pitch_proxy):
    # Proxy tương đối:
    # thấp hơn vùng bình thường -> mũi gần mắt hơn
    # cao hơn vùng bình thường -> mũi gần miệng hơn
    if pitch_proxy < 0.35:
        return "pitch_up_proxy"
    if pitch_proxy > 0.70:
        return "pitch_down_proxy"
    return "pitch_normal_proxy"


def is_roll_tilt(roll_abs):
    return roll_abs >= 18.0


def collect_images(root: Path):
    items = []

    for person in PEOPLE:
        for layer in LAYERS:
            layer_dir = root / person / layer
            if not layer_dir.exists():
                continue

            for p in layer_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    items.append((person, layer, p))

    return items


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=r"D:\warehouse_dataset_v2\face_train_layered",
    )
    parser.add_argument(
        "--out-dir",
        default=r"D:\warehouse_dataset_v2\audit_reports",
    )
    parser.add_argument(
        "--model-name",
        default="buffalo_l",
    )
    parser.add_argument(
        "--det-size",
        default="640,640",
    )
    parser.add_argument(
        "--det-thresh",
        type=float,
        default=0.30,
    )

    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    det_w, det_h = [int(x.strip()) for x in args.det_size.split(",")]
    app = init_face_app(args.model_name, (det_w, det_h), args.det_thresh)

    items = collect_images(root)

    detail_csv = out_dir / "face_pose_coverage_detail.csv"
    summary_csv = out_dir / "face_pose_coverage_summary.csv"
    readiness_csv = out_dir / "face_pose_coverage_readiness.csv"

    counters = {}
    layer_counters = {}

    for person in PEOPLE:
        counters[person] = {
            "total": 0,
            "no_face": 0,
            "front": 0,
            "slight_left_img": 0,
            "slight_right_img": 0,
            "strong_left_img": 0,
            "strong_right_img": 0,
            "roll_tilt": 0,
            "pitch_up_proxy": 0,
            "pitch_down_proxy": 0,
            "pitch_normal_proxy": 0,
        }

        for layer in LAYERS:
            layer_counters[(person, layer)] = {
                "total": 0,
                "no_face": 0,
                "front": 0,
                "slight_left_img": 0,
                "slight_right_img": 0,
                "strong_left_img": 0,
                "strong_right_img": 0,
                "roll_tilt": 0,
                "pitch_up_proxy": 0,
                "pitch_down_proxy": 0,
                "pitch_normal_proxy": 0,
            }

    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "person",
            "layer",
            "filename",
            "path",
            "face_detected",
            "det_score",
            "yaw_signed",
            "yaw_abs",
            "yaw_bin",
            "roll_deg",
            "roll_abs",
            "roll_tilt",
            "pitch_proxy",
            "pitch_bin",
            "bbox",
        ])

        for idx, (person, layer, path) in enumerate(items, start=1):
            img = imread_unicode(path)
            if img is None:
                continue

            faces = app.get(img)
            face = largest_face(faces)

            counters[person]["total"] += 1
            layer_counters[(person, layer)]["total"] += 1

            if face is None:
                counters[person]["no_face"] += 1
                layer_counters[(person, layer)]["no_face"] += 1

                writer.writerow([
                    person, layer, path.name, str(path),
                    0, 0, "", "", "no_face", "", "", "", "", "no_face", "",
                ])
                continue

            pose = calc_pose_proxy(face)

            if pose is None:
                counters[person]["no_face"] += 1
                layer_counters[(person, layer)]["no_face"] += 1

                writer.writerow([
                    person, layer, path.name, str(path),
                    1, float(face.det_score), "", "", "no_kps", "", "", "", "", "no_kps",
                    [int(x) for x in face.bbox],
                ])
                continue

            yaw_bin = classify_yaw(pose["yaw_signed"], pose["yaw_abs"])
            pitch_bin = classify_pitch(pose["pitch_proxy"])
            roll_tilt = is_roll_tilt(pose["roll_abs"])

            counters[person][yaw_bin] += 1
            counters[person][pitch_bin] += 1
            if roll_tilt:
                counters[person]["roll_tilt"] += 1

            layer_counters[(person, layer)][yaw_bin] += 1
            layer_counters[(person, layer)][pitch_bin] += 1
            if roll_tilt:
                layer_counters[(person, layer)]["roll_tilt"] += 1

            writer.writerow([
                person,
                layer,
                path.name,
                str(path),
                1,
                f"{float(face.det_score):.4f}",
                f"{pose['yaw_signed']:.4f}",
                f"{pose['yaw_abs']:.4f}",
                yaw_bin,
                f"{pose['roll_deg']:.2f}",
                f"{pose['roll_abs']:.2f}",
                int(roll_tilt),
                f"{pose['pitch_proxy']:.4f}",
                pitch_bin,
                [int(x) for x in face.bbox],
            ])

            if idx % 100 == 0:
                print(f"Processed {idx}/{len(items)}")

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "person",
            "layer",
            "total",
            "no_face",
            "front",
            "slight_left_img",
            "slight_right_img",
            "strong_left_img",
            "strong_right_img",
            "roll_tilt",
            "pitch_up_proxy",
            "pitch_normal_proxy",
            "pitch_down_proxy",
        ])

        for person in PEOPLE:
            for layer in LAYERS:
                d = layer_counters[(person, layer)]
                writer.writerow([
                    person,
                    layer,
                    d["total"],
                    d["no_face"],
                    d["front"],
                    d["slight_left_img"],
                    d["slight_right_img"],
                    d["strong_left_img"],
                    d["strong_right_img"],
                    d["roll_tilt"],
                    d["pitch_up_proxy"],
                    d["pitch_normal_proxy"],
                    d["pitch_down_proxy"],
                ])

    # Targets để đánh giá đủ góc chưa
    targets = {
        "front": 100,
        "slight_left_img": 30,
        "slight_right_img": 30,
        "strong_left_img": 20,
        "strong_right_img": 20,
        "roll_tilt": 20,
        "pitch_up_proxy": 20,
        "pitch_down_proxy": 20,
    }

    with open(readiness_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "person",
            "angle_type",
            "count",
            "target",
            "status",
            "missing",
        ])

        for person in PEOPLE:
            d = counters[person]

            for angle_type, target in targets.items():
                count = d[angle_type]
                missing = max(0, target - count)
                status = "PASS" if count >= target else "MISSING"

                writer.writerow([
                    person,
                    angle_type,
                    count,
                    target,
                    status,
                    missing,
                ])

    print("=" * 90)
    print("FACE POSE COVERAGE AUDIT DONE")
    print(f"Root: {root}")
    print(f"Total images: {len(items)}")
    print(f"Detail: {detail_csv}")
    print(f"Summary: {summary_csv}")
    print(f"Readiness: {readiness_csv}")
    print("=" * 90)

    for person in PEOPLE:
        d = counters[person]
        print(f"\n===== {person} =====")
        print(f"total: {d['total']}")
        print(f"no_face: {d['no_face']}")
        print(f"front: {d['front']}")
        print(f"slight_left_img: {d['slight_left_img']}")
        print(f"slight_right_img: {d['slight_right_img']}")
        print(f"strong_left_img: {d['strong_left_img']}")
        print(f"strong_right_img: {d['strong_right_img']}")
        print(f"roll_tilt: {d['roll_tilt']}")
        print(f"pitch_up_proxy: {d['pitch_up_proxy']}")
        print(f"pitch_down_proxy: {d['pitch_down_proxy']}")


if __name__ == "__main__":
    main()