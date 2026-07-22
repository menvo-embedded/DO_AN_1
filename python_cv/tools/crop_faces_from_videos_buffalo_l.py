# tools/crop_faces_from_videos_buffalo_l.py

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def variance_of_laplacian(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def expand_bbox(bbox, frame_w, frame_h, scale=1.35):
    x1, y1, x2, y2 = bbox.astype(int)

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    cx = x1 + bw / 2
    cy = y1 + bh / 2

    nw = bw * scale
    nh = bh * scale

    nx1 = int(clamp(cx - nw / 2, 0, frame_w - 1))
    ny1 = int(clamp(cy - nh / 2, 0, frame_h - 1))
    nx2 = int(clamp(cx + nw / 2, 0, frame_w - 1))
    ny2 = int(clamp(cy + nh / 2, 0, frame_h - 1))

    return nx1, ny1, nx2, ny2


def calc_quality(face_crop, det_score, bbox, frame_w, frame_h):
    x1, y1, x2, y2 = bbox.astype(int)

    face_w = max(1, x2 - x1)
    face_h = max(1, y2 - y1)

    face_area_ratio = (face_w * face_h) / max(1, frame_w * frame_h)
    sharpness = variance_of_laplacian(face_crop)

    sharp_norm = min(sharpness / 300.0, 1.0)
    size_norm = min(face_area_ratio / 0.04, 1.0)

    quality = (
        0.45 * float(det_score)
        + 0.35 * sharp_norm
        + 0.20 * size_norm
    )

    return quality, sharpness, face_area_ratio


def init_face_app(model_name, det_size, det_thresh):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    print("=" * 90)
    print("Loading InsightFace")
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


def collect_video_files(input_dir):
    input_dir = Path(input_dir)

    videos = []

    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            videos.append(p)

    return sorted(videos)


def process_video(
    app,
    video_path,
    output_dir,
    emp_id=None,
    sample_every=5,
    max_candidates=500,
    min_det_score=0.35,
    min_face_size=28,
    largest_only=False,
    save_debug=True,
):
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    video_stem = video_path.stem

    if emp_id:
        crop_dir = output_dir / "labeled" / emp_id / video_stem
    else:
        crop_dir = output_dir / "review" / video_stem

    debug_dir = output_dir / "debug_full" / video_stem

    crop_dir.mkdir(parents=True, exist_ok=True)

    if save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    print("\n" + "=" * 90)
    print(f"Video: {video_path}")
    print(f"FPS={fps:.2f}, total_frames={total_frames}")
    print(f"Output: {crop_dir}")
    print("=" * 90)

    candidates = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % sample_every != 0:
            frame_idx += 1
            continue

        h, w = frame.shape[:2]

        faces = app.get(frame)

        if not faces:
            frame_idx += 1
            continue

        if largest_only:
            faces = sorted(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True,
            )[:1]

        for face_i, face in enumerate(faces):
            bbox = face.bbox
            x1, y1, x2, y2 = bbox.astype(int)

            face_w = x2 - x1
            face_h = y2 - y1

            if face.det_score < min_det_score:
                continue

            if face_w < min_face_size or face_h < min_face_size:
                continue

            ex1, ey1, ex2, ey2 = expand_bbox(
                bbox=bbox,
                frame_w=w,
                frame_h=h,
                scale=1.35,
            )

            crop = frame[ey1:ey2, ex1:ex2]

            if crop.size == 0:
                continue

            quality, sharpness, area_ratio = calc_quality(
                face_crop=crop,
                det_score=face.det_score,
                bbox=bbox,
                frame_w=w,
                frame_h=h,
            )

            candidates.append({
                "video": video_path.name,
                "frame_idx": frame_idx,
                "face_i": face_i,
                "det_score": float(face.det_score),
                "quality": float(quality),
                "sharpness": float(sharpness),
                "area_ratio": float(area_ratio),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "expanded_bbox": [int(ex1), int(ey1), int(ex2), int(ey2)],
                "crop": crop,
                "frame": frame.copy() if save_debug else None,
            })

        frame_idx += 1

    cap.release()

    candidates = sorted(
        candidates,
        key=lambda x: x["quality"],
        reverse=True,
    )

    selected = candidates[:max_candidates]

    csv_path = crop_dir / "metadata.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "filename",
            "video",
            "frame_idx",
            "face_i",
            "det_score",
            "quality",
            "sharpness",
            "area_ratio",
            "bbox",
            "expanded_bbox",
        ])

        for rank, item in enumerate(selected, start=1):
            filename = (
                f"{emp_id or 'REVIEW'}_"
                f"{video_stem}_"
                f"rank{rank:04d}_"
                f"frame{item['frame_idx']:06d}_"
                f"q{item['quality']:.3f}_"
                f"s{item['det_score']:.3f}.jpg"
            )

            crop_path = crop_dir / filename
            cv2.imwrite(str(crop_path), item["crop"])

            writer.writerow([
                filename,
                item["video"],
                item["frame_idx"],
                item["face_i"],
                f"{item['det_score']:.4f}",
                f"{item['quality']:.4f}",
                f"{item['sharpness']:.2f}",
                f"{item['area_ratio']:.6f}",
                item["bbox"],
                item["expanded_bbox"],
            ])

            if save_debug and item["frame"] is not None and rank <= 50:
                vis = item["frame"].copy()

                x1, y1, x2, y2 = item["bbox"]
                ex1, ey1, ex2, ey2 = item["expanded_bbox"]

                cv2.rectangle(vis, (ex1, ey1), (ex2, ey2), (255, 180, 0), 2)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(
                    vis,
                    f"q={item['quality']:.3f} det={item['det_score']:.3f}",
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                debug_path = debug_dir / filename.replace(".jpg", "_full.jpg")
                cv2.imwrite(str(debug_path), vis)

    print(f"Candidates found: {len(candidates)}")
    print(f"Selected top: {len(selected)}")
    print(f"Metadata: {csv_path}")

    return selected


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        default=r"D:\FACE",
        help="Folder chứa video",
    )

    parser.add_argument(
        "--output-dir",
        default=r"D:\FACE_OUTPUT",
        help="Folder output",
    )

    parser.add_argument(
        "--emp-id",
        default=None,
        help="NV001/NV002/... Nếu không truyền thì lưu vào review",
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
        "--sample-every",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--min-det-score",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--min-face-size",
        type=int,
        default=28,
    )

    parser.add_argument(
        "--largest-only",
        action="store_true",
        help="Chỉ lấy mặt lớn nhất mỗi frame. Chỉ dùng nếu video chủ yếu 1 người target.",
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

    videos = collect_video_files(args.input_dir)

    if not videos:
        print(f"[ERROR] No videos found in: {args.input_dir}")
        return

    print(f"Found {len(videos)} videos")

    total_selected = 0
    t0 = time.time()

    for video in videos:
        selected = process_video(
            app=app,
            video_path=video,
            output_dir=args.output_dir,
            emp_id=args.emp_id,
            sample_every=args.sample_every,
            max_candidates=args.max_candidates,
            min_det_score=args.min_det_score,
            min_face_size=args.min_face_size,
            largest_only=args.largest_only,
            save_debug=not args.no_debug,
        )

        total_selected += len(selected)

    elapsed = time.time() - t0

    print("\n" + "=" * 90)
    print("DONE")
    print(f"Total selected crops: {total_selected}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Output dir: {args.output_dir}")
    print("=" * 90)


if __name__ == "__main__":
    main()