# tools/classify_face_quality_layers.py

import argparse
import csv
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


LAYER_NAMES = {
    "L1": "L1_easy_front_clear",
    "L2": "L2_normal_angle",
    "L3": "L3_side_angle",
    "L4": "L4_low_light_blur",
    "L5": "L5_hard_review",
}


def variance_of_laplacian(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_contrast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    return brightness, contrast


def collect_images(input_dir):
    input_dir = Path(input_dir)
    images = []

    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images.append(p)

    return sorted(images)


def init_face_app(model_name, det_size, det_thresh):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    print("=" * 90)
    print("Loading InsightFace for quality classification")
    print(f"model_name={model_name}")
    print(f"det_size={det_size}")
    print(f"det_thresh={det_thresh}")
    print(f"providers={providers}")
    print("=" * 90)

    app = FaceAnalysis(
        name=model_name,
        providers=providers,
        allowed_modules=["detection", "recognition"],
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
    Ước lượng đơn giản từ 5 landmarks:
    kps thường gồm: left_eye, right_eye, nose, left_mouth, right_mouth
    Không phải pose thật, nhưng đủ để phân loại ảnh nghiêng/roll tương đối.
    """
    if not hasattr(face, "kps") or face.kps is None or len(face.kps) < 3:
        return 0.0, 0.0

    kps = face.kps.astype(float)

    left_eye = kps[0]
    right_eye = kps[1]
    nose = kps[2]

    eye_center = (left_eye + right_eye) / 2.0
    eye_dist = float(np.linalg.norm(right_eye - left_eye)) + 1e-6

    # Roll: mắt lệch ngang/nghiêng đầu
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    roll_deg = abs(math.degrees(math.atan2(dy, dx)))

    # Yaw proxy: mũi lệch khỏi trung tâm 2 mắt
    yaw_proxy = abs(float(nose[0] - eye_center[0])) / eye_dist

    return yaw_proxy, roll_deg


def is_bbox_near_edge(bbox, w, h, margin_ratio=0.03):
    x1, y1, x2, y2 = bbox.astype(int)

    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)

    return (
        x1 <= mx
        or y1 <= my
        or x2 >= w - mx
        or y2 >= h - my
    )


def score_quality(
    det_score,
    sharpness,
    brightness,
    contrast,
    face_area_ratio,
    yaw_proxy,
    roll_deg,
    edge_cut,
):
    # Normalize các chỉ số
    det_norm = min(max(det_score, 0.0), 1.0)
    sharp_norm = min(sharpness / 80.0, 1.0)
    contrast_norm = min(contrast / 55.0, 1.0)
    size_norm = min(face_area_ratio / 0.30, 1.0)

    # Brightness tốt nhất khoảng 80-190
    if 70 <= brightness <= 200:
        bright_norm = 1.0
    elif 50 <= brightness < 70 or 200 < brightness <= 225:
        bright_norm = 0.7
    else:
        bright_norm = 0.35

    # Góc càng nhỏ càng tốt cho L1/L2
    yaw_penalty = min(yaw_proxy / 0.80, 1.0)
    roll_penalty = min(roll_deg / 35.0, 1.0)

    score = (
        0.30 * det_norm
        + 0.25 * sharp_norm
        + 0.15 * contrast_norm
        + 0.15 * size_norm
        + 0.10 * bright_norm
        - 0.03 * yaw_penalty
        - 0.02 * roll_penalty
    )

    if edge_cut:
        score -= 0.12

    return float(max(0.0, min(1.0, score)))


def classify_layer(
    det_score,
    quality_score,
    sharpness,
    brightness,
    contrast,
    face_area_ratio,
    yaw_proxy,
    roll_deg,
    edge_cut,
    no_face=False,
):
    if no_face:
        return "L5"

    # L1: ảnh rất sạch, nên dùng gallery chính
    if (
        det_score >= 0.80
        and quality_score >= 0.62
        and sharpness >= 25
        and 65 <= brightness <= 210
        and contrast >= 20
        and face_area_ratio >= 0.12
        and yaw_proxy <= 0.35
        and roll_deg <= 18
        and not edge_cut
    ):
        return "L1"

    # L2: ảnh tốt, hơi nghiêng nhẹ
    if (
        det_score >= 0.72
        and quality_score >= 0.55
        and sharpness >= 18
        and 50 <= brightness <= 225
        and contrast >= 15
        and face_area_ratio >= 0.07
        and yaw_proxy <= 0.55
    ):
        return "L2"

    # L3: góc khó hơn nhưng còn dùng được
    if (
        det_score >= 0.60
        and quality_score >= 0.45
        and sharpness >= 10
        and face_area_ratio >= 0.04
    ):
        return "L3"

    # L4: ảnh yếu, để test, chưa đưa gallery chính
    if det_score >= 0.40:
        return "L4"

    return "L5"


def annotate_debug(img, face, metrics_text):
    vis = img.copy()
    h, w = vis.shape[:2]

    if face is not None:
        x1, y1, x2, y2 = face.bbox.astype(int)
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if hasattr(face, "kps") and face.kps is not None:
            for p in face.kps.astype(int):
                cv2.circle(vis, tuple(p), 3, (0, 255, 255), -1)

    y = 25
    for line in metrics_text:
        cv2.putText(
            vis,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
        y += 24

    return vis


def process_images(
    app,
    input_dir,
    output_dir,
    person_name,
    copy_mode=True,
    save_debug=True,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    person_dir = output_dir / person_name
    debug_dir = output_dir / "_debug" / person_name

    for layer_name in LAYER_NAMES.values():
        (person_dir / layer_name).mkdir(parents=True, exist_ok=True)

    if save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_dir)

    if not images:
        print(f"[ERROR] No images found in: {input_dir}")
        return

    print(f"Found images: {len(images)}")
    print(f"Output person dir: {person_dir}")

    csv_path = person_dir / "quality_report.csv"

    counts = {k: 0 for k in LAYER_NAMES.keys()}

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "filename",
            "src_path",
            "layer",
            "quality_score",
            "det_score",
            "sharpness",
            "brightness",
            "contrast",
            "face_area_ratio",
            "yaw_proxy",
            "roll_deg",
            "edge_cut",
            "img_w",
            "img_h",
            "bbox",
        ])

        for idx, img_path in enumerate(images, start=1):
            img = cv2.imread(str(img_path))

            if img is None:
                continue

            h, w = img.shape[:2]

            sharpness = variance_of_laplacian(img)
            brightness, contrast = brightness_contrast(img)

            faces = app.get(img)
            face = largest_face(faces)

            no_face = face is None

            if no_face:
                det_score = 0.0
                face_area_ratio = 0.0
                yaw_proxy = 0.0
                roll_deg = 0.0
                edge_cut = True
                quality_score = 0.0
                bbox = []
                layer = "L5"
            else:
                bbox_arr = face.bbox
                x1, y1, x2, y2 = bbox_arr.astype(int)
                face_area_ratio = max(1, (x2 - x1) * (y2 - y1)) / max(1, w * h)

                det_score = float(face.det_score)
                yaw_proxy, roll_deg = calc_pose_proxy(face)
                edge_cut = is_bbox_near_edge(bbox_arr, w, h)

                quality_score = score_quality(
                    det_score=det_score,
                    sharpness=sharpness,
                    brightness=brightness,
                    contrast=contrast,
                    face_area_ratio=face_area_ratio,
                    yaw_proxy=yaw_proxy,
                    roll_deg=roll_deg,
                    edge_cut=edge_cut,
                )

                layer = classify_layer(
                    det_score=det_score,
                    quality_score=quality_score,
                    sharpness=sharpness,
                    brightness=brightness,
                    contrast=contrast,
                    face_area_ratio=face_area_ratio,
                    yaw_proxy=yaw_proxy,
                    roll_deg=roll_deg,
                    edge_cut=edge_cut,
                    no_face=False,
                )

                bbox = [int(x1), int(y1), int(x2), int(y2)]

            counts[layer] += 1

            dst_layer_dir = person_dir / LAYER_NAMES[layer]

            new_name = (
                f"{layer}_"
                f"q{quality_score:.3f}_"
                f"d{det_score:.3f}_"
                f"sharp{sharpness:.0f}_"
                f"{img_path.name}"
            )

            dst_path = dst_layer_dir / new_name

            if copy_mode:
                shutil.copy2(img_path, dst_path)
            else:
                shutil.move(str(img_path), str(dst_path))

            writer.writerow([
                new_name,
                str(img_path),
                layer,
                f"{quality_score:.4f}",
                f"{det_score:.4f}",
                f"{sharpness:.2f}",
                f"{brightness:.2f}",
                f"{contrast:.2f}",
                f"{face_area_ratio:.6f}",
                f"{yaw_proxy:.4f}",
                f"{roll_deg:.2f}",
                int(edge_cut),
                w,
                h,
                bbox,
            ])

            if save_debug and idx <= 300:
                metrics_text = [
                    f"{layer} q={quality_score:.3f} det={det_score:.3f}",
                    f"sharp={sharpness:.1f} bright={brightness:.1f} contrast={contrast:.1f}",
                    f"area={face_area_ratio:.3f} yaw={yaw_proxy:.2f} roll={roll_deg:.1f} edge={edge_cut}",
                ]

                vis = annotate_debug(img, face, metrics_text)
                debug_path = debug_dir / new_name
                cv2.imwrite(str(debug_path), vis)

            if idx % 100 == 0:
                print(f"Processed {idx}/{len(images)}")

    print("\n" + "=" * 90)
    print("DONE QUALITY CLASSIFICATION")
    print(f"Report: {csv_path}")
    print("Counts:")
    for k in ["L1", "L2", "L3", "L4", "L5"]:
        print(f"  {k} {LAYER_NAMES[k]}: {counts[k]}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder chứa ảnh crop face",
    )

    parser.add_argument(
        "--output-dir",
        default=r"D:\warehouse_dataset\face_dataset_layered",
        help="Folder output layered dataset",
    )

    parser.add_argument(
        "--person-name",
        required=True,
        help="VD: NV005_Men hoặc Duc_review",
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
        default=0.35,
    )

    parser.add_argument(
        "--move",
        action="store_true",
        help="Move file thay vì copy",
    )

    parser.add_argument(
        "--no-debug",
        action="store_true",
    )

    args = parser.parse_args()

    det_w, det_h = [int(x.strip()) for x in args.det_size.split(",")]
    det_size = (det_w, det_h)

    app = init_face_app(
        model_name=args.model_name,
        det_size=det_size,
        det_thresh=args.det_thresh,
    )

    process_images(
        app=app,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        person_name=args.person_name,
        copy_mode=not args.move,
        save_debug=not args.no_debug,
    )


if __name__ == "__main__":
    main()