import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from supervision import Detections
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import CAM_ZONE2_RTSP, YOLO_CLASSES, YOLO_CONF, YOLO_WEIGHTS
from reid.gallery import Gallery
from reid.reid_engine import ReIDEngine
from tracking.tracker import Tracker


def open_rtsp():
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(CAM_ZONE2_RTSP, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
    if not cap.isOpened():
        raise RuntimeError("Cannot open Zone2 RTSP")
    return cap


def rank_crop(reid, gallery_data, crop):
    emb = reid.get_embedding(crop)
    if emb is None:
        return []

    rows = []
    for emp_id, embeds in gallery_data.items():
        score = reid.match_score(emb, embeds)
        rows.append((emp_id, score))

    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture live Zone2 person crops and Re-ID rankings."
    )
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--save-every", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "outputs" / "zone2_reid_debug")
    return parser.parse_args()


def main():
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / run_id
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "zone2_reid_rankings.csv"

    print("=" * 80)
    print("ZONE2 RE-ID DEBUG CAPTURE")
    print("=" * 80)
    print("Output:", out_dir)
    print("Duration:", args.duration)
    print("Save every:", args.save_every)

    reid = ReIDEngine()
    gallery = Gallery()
    gallery_data = gallery.all()
    model = YOLO(str(YOLO_WEIGHTS))
    tracker = Tracker()
    cap = open_rtsp()

    last_saved = {}
    start = time.time()
    saved_count = 0

    fieldnames = [
        "timestamp",
        "track_id",
        "crop_path",
        "bbox",
        "best_id",
        "best_score",
        "second_id",
        "second_score",
        "margin",
        "ranking",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        while time.time() - start < args.duration:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            frame = cv2.resize(frame, (640, 480))
            results = model(
                frame,
                classes=YOLO_CLASSES,
                conf=YOLO_CONF,
                verbose=False,
            )[0]
            dets = Detections.from_ultralytics(results)
            tracked = tracker.update(dets)

            h, w = frame.shape[:2]
            track_ids = tracked.tracker_id if tracked.tracker_id is not None else []
            now = time.time()

            for i, track_id in enumerate(track_ids):
                last_ts = last_saved.get(int(track_id), 0.0)
                if now - last_ts < args.save_every:
                    continue

                x1, y1, x2, y2 = tracked.xyxy[i].astype(int)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                ranking = rank_crop(reid, gallery_data, crop)
                if not ranking:
                    continue

                best_id, best_score = ranking[0]
                second_id, second_score = ranking[1] if len(ranking) > 1 else ("", 0.0)
                margin = best_score - second_score

                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                crop_name = f"zone2_track{int(track_id)}_{ts}_{best_id}_{best_score:.3f}.jpg"
                crop_path = crops_dir / crop_name
                cv2.imwrite(str(crop_path), crop)

                writer.writerow({
                    "timestamp": ts,
                    "track_id": int(track_id),
                    "crop_path": str(crop_path),
                    "bbox": f"{x1},{y1},{x2},{y2}",
                    "best_id": best_id,
                    "best_score": f"{best_score:.6f}",
                    "second_id": second_id,
                    "second_score": f"{second_score:.6f}",
                    "margin": f"{margin:.6f}",
                    "ranking": ";".join(f"{emp}:{score:.6f}" for emp, score in ranking),
                })
                f.flush()

                saved_count += 1
                last_saved[int(track_id)] = now
                print(
                    f"saved={saved_count:03d} track={int(track_id)} "
                    f"best={best_id} score={best_score:.4f} "
                    f"second={second_id} second_score={second_score:.4f} "
                    f"margin={margin:.4f}"
                )

    cap.release()
    print()
    print("Saved crops:", crops_dir)
    print("Saved CSV:", csv_path)


if __name__ == "__main__":
    main()
