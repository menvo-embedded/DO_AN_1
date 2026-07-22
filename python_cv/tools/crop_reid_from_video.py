from pathlib import Path
import argparse
import csv
import sys
import cv2
import numpy as np
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import DATASET_CROPS_ROOT

# =========================
# QUALITY CONFIG DEFAULT
# =========================

DEFAULT_OUTPUT_ROOT = str(DATASET_CROPS_ROOT)

IMAGE_EXTS = [".mp4", ".mov", ".avi", ".mkv", ".m4v"]


def find_video_by_name(video_name: str, search_root: str = "D:\\") -> Path:
    """
    Tìm video theo tên/stem trên ổ D.
    Ví dụ nhập VID_20260507_014519 thì tự tìm VID_20260507_014519.mp4/.mov...
    """
    p = Path(video_name)

    if p.exists():
        return p

    search_root = Path(search_root)

    # Nếu user nhập có đuôi file
    if p.suffix:
        matches = list(search_root.rglob(p.name))
    else:
        matches = []
        for ext in IMAGE_EXTS:
            matches.extend(search_root.rglob(video_name + ext))

    if not matches:
        raise FileNotFoundError(f"Không tìm thấy video: {video_name} trong {search_root}")

    matches = sorted(matches, key=lambda x: x.stat().st_mtime, reverse=True)
    return matches[0]


def rotate_frame(frame, rotate: str):
    if rotate == "none":
        return frame
    if rotate == "cw":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotate == "ccw":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotate == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def laplacian_blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def brightness_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def expand_box(x1, y1, x2, y2, w, h, pad_ratio=0.08):
    bw = x2 - x1
    bh = y2 - y1

    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)

    nx1 = max(0, x1 - px)
    ny1 = max(0, y1 - py)
    nx2 = min(w - 1, x2 + px)
    ny2 = min(h - 1, y2 + py)

    return nx1, ny1, nx2, ny2


def is_near_edge(x1, y1, x2, y2, w, h, edge_margin=5):
    return x1 <= edge_margin or y1 <= edge_margin or x2 >= w - edge_margin or y2 >= h - edge_margin


def quality_check(
    crop,
    bbox,
    frame_shape,
    conf,
    min_conf,
    min_w,
    min_h,
    min_area_ratio,
    max_area_ratio,
    min_aspect,
    max_aspect,
    min_blur,
    min_brightness,
    max_brightness,
    reject_edge_touch,
):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1

    area_ratio = (bw * bh) / float(w * h)
    aspect = bw / float(bh + 1e-6)
    blur = laplacian_blur_score(crop)
    bright = brightness_score(crop)

    reasons = []

    if conf < min_conf:
        reasons.append(f"low_conf:{conf:.2f}")

    if bw < min_w:
        reasons.append(f"small_w:{bw}")

    if bh < min_h:
        reasons.append(f"small_h:{bh}")

    if area_ratio < min_area_ratio:
        reasons.append(f"small_area:{area_ratio:.3f}")

    if area_ratio > max_area_ratio:
        reasons.append(f"too_large_area:{area_ratio:.3f}")

    if aspect < min_aspect or aspect > max_aspect:
        reasons.append(f"bad_aspect:{aspect:.2f}")

    if blur < min_blur:
        reasons.append(f"blur:{blur:.1f}")

    if bright < min_brightness:
        reasons.append(f"dark:{bright:.1f}")

    if bright > max_brightness:
        reasons.append(f"bright:{bright:.1f}")

    if reject_edge_touch and is_near_edge(x1, y1, x2, y2, w, h):
        reasons.append("touch_edge")

    ok = len(reasons) == 0

    info = {
        "conf": conf,
        "bbox_w": bw,
        "bbox_h": bh,
        "area_ratio": area_ratio,
        "aspect": aspect,
        "blur": blur,
        "brightness": bright,
        "reasons": "|".join(reasons),
    }

    return ok, info


def draw_box(frame, bbox, text, color):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        text,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", type=str, default=None, help="Đường dẫn video cụ thể.")
    parser.add_argument("--video-name", type=str, default="VID_20260507_014519", help="Tên video để tìm trong ổ D.")
    parser.add_argument("--search-root", type=str, default="D:\\", help="Ổ/thư mục để tìm video.")

    parser.add_argument("--emp-id", type=str, required=True, help="Mã nhân viên/person, ví dụ NV005 hoặc NV006.")
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="YOLO weight, ví dụ yolo11n.pt hoặc yolo11n-seg.pt.")
    parser.add_argument("--rotate", type=str, default="none", choices=["none", "cw", "ccw", "180"])

    parser.add_argument("--sample-every", type=int, default=5, help="Lấy mỗi N frame để tránh quá nhiều ảnh giống nhau.")
    parser.add_argument("--max-crops", type=int, default=300)
    parser.add_argument("--display", action="store_true")

    parser.add_argument("--skip-multi-person", action="store_true", help="Nếu frame có nhiều người thì bỏ qua để data sạch.")
    parser.add_argument("--select-largest", action="store_true", help="Nếu nhiều người thì lấy bbox lớn nhất.")

    parser.add_argument("--min-conf", type=float, default=0.55)
    parser.add_argument("--min-w", type=int, default=70)
    parser.add_argument("--min-h", type=int, default=120)
    parser.add_argument("--min-area-ratio", type=float, default=0.035)
    parser.add_argument("--max-area-ratio", type=float, default=0.70)
    parser.add_argument("--min-aspect", type=float, default=0.25)
    parser.add_argument("--max-aspect", type=float, default=0.95)
    parser.add_argument("--min-blur", type=float, default=35.0)
    parser.add_argument("--min-brightness", type=float, default=35.0)
    parser.add_argument("--max-brightness", type=float, default=230.0)
    parser.add_argument("--pad-ratio", type=float, default=0.08)
    parser.add_argument("--reject-edge-touch", action="store_true")

    args = parser.parse_args()

    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            raise FileNotFoundError(video_path)
    else:
        video_path = find_video_by_name(args.video_name, args.search_root)

    print("=" * 100)
    print("CROP RE-ID DATA FROM VIDEO")
    print("=" * 100)
    print("Video:", video_path)
    print("Employee ID:", args.emp_id)
    print("Output root:", args.output_root)
    print("YOLO weights:", args.weights)
    print("=" * 100)

    model = YOLO(args.weights)

    output_dir = Path(args.output_root) / args.emp_id / video_path.stem
    crop_dir = output_dir / "crops"
    annotated_dir = output_dir / "annotated"
    reject_dir = output_dir / "rejected"

    crop_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    reject_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "crop_log.csv"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    frame_idx = 0
    saved = 0
    rejected = 0

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_idx",
                "status",
                "filename",
                "conf",
                "bbox",
                "bbox_w",
                "bbox_h",
                "area_ratio",
                "aspect",
                "blur",
                "brightness",
                "reasons",
            ],
        )
        writer.writeheader()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            if frame_idx % args.sample_every != 0:
                continue

            frame = rotate_frame(frame, args.rotate)
            fh, fw = frame.shape[:2]

            result = model(frame, classes=[0], conf=args.min_conf, verbose=False)[0]

            detections = []

            if result.boxes is not None:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    if cls != 0:
                        continue

                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()

                    x1 = max(0, min(x1, fw - 1))
                    y1 = max(0, min(y1, fh - 1))
                    x2 = max(0, min(x2, fw - 1))
                    y2 = max(0, min(y2, fh - 1))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    area = (x2 - x1) * (y2 - y1)

                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "conf": conf,
                        "area": area,
                    })

            if len(detections) == 0:
                continue

            if args.skip_multi_person and len(detections) > 1:
                annotated = frame.copy()
                for d in detections:
                    draw_box(annotated, d["bbox"], f"SKIP multi {d['conf']:.2f}", (0, 165, 255))

                reject_name = f"reject_multi_frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(reject_dir / reject_name), annotated)

                writer.writerow({
                    "frame_idx": frame_idx,
                    "status": "reject_multi_person",
                    "filename": reject_name,
                    "conf": "",
                    "bbox": "",
                    "bbox_w": "",
                    "bbox_h": "",
                    "area_ratio": "",
                    "aspect": "",
                    "blur": "",
                    "brightness": "",
                    "reasons": f"multi_person:{len(detections)}",
                })

                rejected += 1
                continue

            if args.select_largest:
                detections = [max(detections, key=lambda d: d["area"])]

            annotated = frame.copy()

            for det_i, det in enumerate(detections):
                x1, y1, x2, y2 = det["bbox"]
                conf = det["conf"]

                ex1, ey1, ex2, ey2 = expand_box(
                    x1, y1, x2, y2,
                    fw, fh,
                    pad_ratio=args.pad_ratio,
                )

                crop = frame[ey1:ey2, ex1:ex2].copy()

                if crop.size == 0:
                    continue

                ok, info = quality_check(
                    crop=crop,
                    bbox=(x1, y1, x2, y2),
                    frame_shape=frame.shape,
                    conf=conf,
                    min_conf=args.min_conf,
                    min_w=args.min_w,
                    min_h=args.min_h,
                    min_area_ratio=args.min_area_ratio,
                    max_area_ratio=args.max_area_ratio,
                    min_aspect=args.min_aspect,
                    max_aspect=args.max_aspect,
                    min_blur=args.min_blur,
                    min_brightness=args.min_brightness,
                    max_brightness=args.max_brightness,
                    reject_edge_touch=args.reject_edge_touch,
                )

                if ok:
                    saved += 1
                    filename = f"{args.emp_id}_{video_path.stem}_f{frame_idx:06d}_{saved:04d}.jpg"
                    cv2.imwrite(str(crop_dir / filename), crop)

                    draw_box(
                        annotated,
                        (x1, y1, x2, y2),
                        f"KEEP {args.emp_id} {conf:.2f}",
                        (0, 255, 0),
                    )

                    writer.writerow({
                        "frame_idx": frame_idx,
                        "status": "keep",
                        "filename": filename,
                        "conf": f"{conf:.4f}",
                        "bbox": f"{x1},{y1},{x2},{y2}",
                        "bbox_w": info["bbox_w"],
                        "bbox_h": info["bbox_h"],
                        "area_ratio": f"{info['area_ratio']:.4f}",
                        "aspect": f"{info['aspect']:.4f}",
                        "blur": f"{info['blur']:.2f}",
                        "brightness": f"{info['brightness']:.2f}",
                        "reasons": "",
                    })

                else:
                    rejected += 1
                    filename = f"reject_{args.emp_id}_{video_path.stem}_f{frame_idx:06d}_{rejected:04d}.jpg"
                    cv2.imwrite(str(reject_dir / filename), crop)

                    draw_box(
                        annotated,
                        (x1, y1, x2, y2),
                        f"REJECT {info['reasons']}",
                        (0, 0, 255),
                    )

                    writer.writerow({
                        "frame_idx": frame_idx,
                        "status": "reject_quality",
                        "filename": filename,
                        "conf": f"{conf:.4f}",
                        "bbox": f"{x1},{y1},{x2},{y2}",
                        "bbox_w": info["bbox_w"],
                        "bbox_h": info["bbox_h"],
                        "area_ratio": f"{info['area_ratio']:.4f}",
                        "aspect": f"{info['aspect']:.4f}",
                        "blur": f"{info['blur']:.2f}",
                        "brightness": f"{info['brightness']:.2f}",
                        "reasons": info["reasons"],
                    })

            if saved > 0 and saved % 20 == 0:
                annotated_name = f"annotated_f{frame_idx:06d}_saved{saved:04d}.jpg"
                cv2.imwrite(str(annotated_dir / annotated_name), annotated)

            if args.display:
                cv2.imshow("crop_reid_from_video", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if saved >= args.max_crops:
                print(f"Đã đạt max_crops={args.max_crops}, dừng.")
                break

    cap.release()
    cv2.destroyAllWindows()

    print("\nDONE")
    print("Saved crops:", saved)
    print("Rejected:", rejected)
    print("Output dir:", output_dir)
    print("Crops:", crop_dir)
    print("Annotated:", annotated_dir)
    print("Rejected:", reject_dir)
    print("CSV log:", csv_path)


if __name__ == "__main__":
    main()
