# tests/test_webcam_reid_live.py
# Test nhanh bằng webcam ngoài quán cafe:
# Webcam -> YOLO person -> ByteTrack -> Body Re-ID -> Dashboard presence

import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from supervision import Detections

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import YOLO_WEIGHTS, YOLO_CONF, YOLO_CLASSES, FLASK_PORT
from tracking.tracker import Tracker
from reid.reid_engine import ReIDEngine
from reid.gallery import Gallery
from database.database import Database
from dashboard.app import run_dashboard
from utils.logger import get_logger


# ============================================================
# CONFIG TEST
# ============================================================

CAM_INDEX = 1
# Webcam laptop thường là 0.
# Nếu không lên hình, đổi thành 1, 2, 3.

FRAME_W = 640
FRAME_H = 480

TEST_ZONE_NAME = "webcam"
TEST_ZONE_NUMBER = 1

IDENTIFY_COOLDOWN_SEC = 2.0

WINDOW_NAME = "WEBCAM REID LIVE TEST"

COLOR_OK = (0, 220, 80)
COLOR_UNKNOWN = (0, 100, 255)


log = get_logger("webcam_test")


def open_camera(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index={index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


def main():
    log.info("=" * 50)
    log.info("WEBCAM REID LIVE TEST START")
    log.info("=" * 50)

    db = Database()
    reid = ReIDEngine()
    gallery = Gallery()

    if not gallery.employees():
        log.warning("Gallery is empty. Run main.py once to preload gallery first.")
    else:
        log.info(f"Gallery loaded: {gallery.employees()}")

    # Dashboard riêng cho test
    t_dash = threading.Thread(
        target=run_dashboard,
        args=(db,),
        name="dashboard",
        daemon=True,
    )
    t_dash.start()
    log.info(f"Dashboard: http://localhost:{FLASK_PORT}")

    model = YOLO(str(YOLO_WEIGHTS))
    tracker = Tracker()

    cap = open_camera(CAM_INDEX)
    log.info(f"Webcam opened: index={CAM_INDEX}")

    confirmed = {}
    last_identify_time = {}

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            log.warning("Frame read failed")
            time.sleep(0.03)
            continue

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
            x1, y1, x2, y2 = tracked.xyxy[i].astype(int)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            track_key = f"{TEST_ZONE_NAME}:{track_id}"

            if track_key not in confirmed:
                last_t = last_identify_time.get(track_key, 0)

                if now - last_t >= IDENTIFY_COOLDOWN_SEC:
                    last_identify_time[track_key] = now

                    crop = frame[y1:y2, x1:x2]

                    if crop.size > 0:
                        emp_id = reid.identify(crop, gallery.all())

                        if emp_id:
                            confirmed[track_key] = emp_id
                            log.info(f"WEBCAM REID: {track_key} -> {emp_id}")
                            db.update_presence(emp_id, TEST_ZONE_NUMBER, track_key)

            label = confirmed.get(track_key, track_key)
            color = COLOR_OK if track_key in confirmed else COLOR_UNKNOWN

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

        dt = now - prev_time
        prev_time = now

        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        cv2.putText(
            frame,
            f"WEBCAM REID TEST | FPS: {fps:.1f} | Q: quit",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key in [ord("q"), ord("Q"), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()
    db.close()


if __name__ == "__main__":
    main()