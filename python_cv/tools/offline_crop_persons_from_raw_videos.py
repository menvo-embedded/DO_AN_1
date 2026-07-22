# tools/offline_crop_persons_from_raw_videos.py
# Offline crop person từ raw video đã thu.
#
# Input:
#   RAW_VIDEO_ROOT
#
# Output review riêng:
#   REVIEW_CROPS_FROM_RAW_ROOT
#
# Mục tiêu:
# - Không gán nhãn tự động.
# - Không đụng dataset chính.
# - Crop tất cả người phát hiện được bằng YOLO.
# - Lưu metadata.csv để truy vết video/frame/zone.
#
# Test 1 session:
#   D:\UV4\anaconda3\python.exe .\tools\offline_crop_persons_from_raw_videos.py --session raw_3zone_test_20s --every 5 --max-frames 200
#
# Chạy toàn bộ:
#   D:\UV4\anaconda3\python.exe .\tools\offline_crop_persons_from_raw_videos.py --all --every 25

import sys
import csv
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


# ============================================================
# PROJECT PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import RAW_VIDEO_ROOT, REVIEW_CROPS_FROM_RAW_ROOT, YOLO_WEIGHTS


# ============================================================
# CONFIG
# ============================================================

RAW_ROOT = RAW_VIDEO_ROOT
OUTPUT_ROOT = REVIEW_CROPS_FROM_RAW_ROOT

VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".mov"}

PERSON_CLASS_ID = 0

YOLO_CONF = 0.35
YOLO_IMGSZ = 960

MIN_CROP_W = 40
MIN_CROP_H = 80
MIN_AREA = 3000

PAD_RATIO = 0.08

SAVE_FULL_DEBUG_EVERY = 0
# 0 = không lưu full frame debug.
# Ví dụ 200 = mỗi 200 frame lưu 1 ảnh full frame có bbox.

JPEG_QUALITY = 95


def parse_zone(video_path: Path) -> str:
    name = video_path.name.lower()

    if "zone1" in name:
        return "zone1"

    if "zone2" in name:
        return "zone2"

    if "zone3" in name:
        return "zone3"

    return "unknown_zone"


def safe_crop(frame, x1, y1, x2, y2, pad_ratio=0.08):
    h, w = frame.shape[:2]

    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = frame[y1:y2, x1:x2]

    if crop is None or crop.size == 0:
        return None, None

    return crop, (x1, y1, x2, y2)


def list_videos(session: str | None, all_sessions: bool):
    if all_sessions:
        return sorted([
            p for p in RAW_ROOT.rglob("*")
            if p.suffix.lower() in VIDEO_EXTS
        ])

    if session is None:
        raise ValueError("Cần truyền --session hoặc dùng --all")

    session_dir = RAW_ROOT / session

    if not session_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy session: {session_dir}")

    return sorted([
        p for p in session_dir.rglob("*")
        if p.suffix.lower() in VIDEO_EXTS
    ])


def draw_debug(frame, boxes):
    out = frame.copy()

    for b in boxes:
        x1, y1, x2, y2, conf = b
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"person {conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return out


def process_video(model, video_path: Path, out_root: Path, every: int, max_frames: int | None):
    zone = parse_zone(video_path)
    session_name = video_path.parent.name
    video_stem = video_path.stem

    video_out_root = out_root / session_name / zone / video_stem
    out_dir = video_out_root / "person_crops"
    debug_dir = video_out_root / "debug_frames"
    metadata_path = video_out_root / "metadata.csv"

    out_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_FULL_DEBUG_EVERY > 0:
        debug_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[FAIL] Cannot open video: {video_path}")
        return {
            "video": str(video_path),
            "frames": 0,
            "saved": 0,
            "status": "open_failed",
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print()
    print("========== PROCESS VIDEO ==========")
    print("Video :", video_path)
    print("Zone  :", zone)
    print("Size  :", f"{width}x{height}")
    print("FPS   :", fps)
    print("Frames:", total_frames)
    print("Every :", every)
    print("Out   :", out_dir)
    print("===================================")

    frame_idx = 0
    saved_count = 0
    read_count = 0

    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "crop_file",
            "session",
            "zone",
            "video_file",
            "frame_idx",
            "time_sec",
            "person_idx",
            "conf",
            "x1",
            "y1",
            "x2",
            "y2",
            "crop_w",
            "crop_h",
            "source_w",
            "source_h",
        ])

        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                break

            read_count += 1

            if max_frames is not None and frame_idx >= max_frames:
                break

            if frame_idx % every != 0:
                frame_idx += 1
                continue

            results = model(
                frame,
                classes=[PERSON_CLASS_ID],
                conf=YOLO_CONF,
                imgsz=YOLO_IMGSZ,
                verbose=False,
            )[0]

            boxes_for_debug = []

            if results.boxes is not None and len(results.boxes) > 0:
                xyxy = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                cls = results.boxes.cls.cpu().numpy()

                person_idx = 0

                for box, conf, c in zip(xyxy, confs, cls):
                    if int(c) != PERSON_CLASS_ID:
                        continue

                    x1, y1, x2, y2 = box.astype(int).tolist()

                    crop_w = x2 - x1
                    crop_h = y2 - y1
                    area = crop_w * crop_h

                    if crop_w < MIN_CROP_W or crop_h < MIN_CROP_H or area < MIN_AREA:
                        continue

                    crop, padded_box = safe_crop(
                        frame,
                        x1,
                        y1,
                        x2,
                        y2,
                        PAD_RATIO,
                    )

                    if crop is None:
                        continue

                    px1, py1, px2, py2 = padded_box
                    crop_h2, crop_w2 = crop.shape[:2]

                    time_sec = frame_idx / fps if fps and fps > 0 else 0

                    crop_name = (
                        f"{zone}_{video_stem}_"
                        f"f{frame_idx:08d}_"
                        f"t{time_sec:08.2f}_"
                        f"p{person_idx:02d}_"
                        f"conf{float(conf):.2f}.jpg"
                    )

                    crop_path = out_dir / crop_name

                    ok = cv2.imwrite(
                        str(crop_path),
                        crop,
                        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
                    )

                    if not ok:
                        continue

                    writer.writerow([
                        str(crop_path),
                        session_name,
                        zone,
                        video_path.name,
                        frame_idx,
                        round(time_sec, 3),
                        person_idx,
                        round(float(conf), 4),
                        px1,
                        py1,
                        px2,
                        py2,
                        crop_w2,
                        crop_h2,
                        width,
                        height,
                    ])

                    boxes_for_debug.append((px1, py1, px2, py2, float(conf)))

                    saved_count += 1
                    person_idx += 1

            if SAVE_FULL_DEBUG_EVERY > 0 and frame_idx % SAVE_FULL_DEBUG_EVERY == 0:
                dbg = draw_debug(frame, boxes_for_debug)
                dbg_path = debug_dir / f"debug_f{frame_idx:08d}.jpg"
                cv2.imwrite(str(dbg_path), dbg)

            if frame_idx % (every * 50) == 0:
                print(
                    f"[PROGRESS] {video_path.name} "
                    f"frame={frame_idx}/{total_frames} saved={saved_count}"
                )

            frame_idx += 1

    cap.release()

    print(f"[DONE] {video_path.name} | read={read_count} | saved={saved_count}")

    return {
        "video": str(video_path),
        "frames": read_count,
        "saved": saved_count,
        "status": "ok",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Tên session trong raw_videos_full",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Xử lý toàn bộ session/video",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=10,
        help="Lấy 1 frame mỗi N frame",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Giới hạn số frame/video để test",
    )

    args = parser.parse_args()

    print("========== OFFLINE PERSON CROP ==========")
    print("RAW_ROOT   :", RAW_ROOT)
    print("OUTPUT_ROOT:", OUTPUT_ROOT)
    print("YOLO       :", YOLO_WEIGHTS)
    print("CONF       :", YOLO_CONF)
    print("IMGSZ      :", YOLO_IMGSZ)
    print("EVERY      :", args.every)
    print("=========================================")

    videos = list_videos(args.session, args.all)

    print(f"[INFO] Found {len(videos)} videos")

    if not videos:
        return

    model = YOLO(str(YOLO_WEIGHTS))

    summary = []

    for video_path in videos:
        try:
            s = process_video(
                model=model,
                video_path=video_path,
                out_root=OUTPUT_ROOT,
                every=max(1, args.every),
                max_frames=args.max_frames,
            )
            summary.append(s)

        except Exception as e:
            print(f"[ERROR] {video_path}: {e}")
            summary.append({
                "video": str(video_path),
                "frames": 0,
                "saved": 0,
                "status": f"error: {e}",
            })

    print()
    print("========== SUMMARY ==========")

    total_saved = 0

    for s in summary:
        print(f"{s['status']:12s} | saved={s['saved']:6d} | {s['video']}")
        total_saved += int(s["saved"])

    print("-----------------------------")
    print("TOTAL SAVED:", total_saved)
    print("OUTPUT:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
