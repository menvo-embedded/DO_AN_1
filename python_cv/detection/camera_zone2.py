# python_cv/detection/camera_zone2.py

import os
import time
import threading
from datetime import datetime

import cv2
import numpy as np
import torch
from supervision import Detections
from ultralytics import YOLO

from config.settings import (
    CAM_ZONE2_TYPE,
    CAM_ZONE2_INDEX,
    CAM_ZONE2_RTSP,
    YOLO_WEIGHTS,
    YOLO_CONF,
    YOLO_CLASSES,
)
from tracking.tracker import Tracker
from fusion.fusion_layer import FusionLayer
from utils.logger import get_logger


log = get_logger("zone2")

COLOR_CONFIRMED = (0, 220, 80)
COLOR_UNKNOWN   = (0, 100, 255)
COLOR_REVIEW    = (0, 200, 255)   # body score cao nhưng margin thấp


# ============================================================
# REALTIME TUNING CONFIG
# ============================================================

YOLO_IMGSZ      = 320
REID_INTERVAL_SEC = 3.0
ENABLE_ZONE2_REID = True    # Bật Re-ID
RTSP_TRANSPORT  = "udp"
SHOW_FPS        = True


class LatestRTSPReader:
    """
    Thread đọc RTSP riêng.
    Luôn giữ frame mới nhất, không xếp hàng frame cũ.
    """
    def __init__(self, open_func):
        self.open_func       = open_func
        self.cap             = None
        self.latest_frame    = None
        self.lock            = threading.Lock()
        self.running         = False
        self.thread          = None
        self.last_read_time  = 0.0

    def start(self):
        self.cap = self.open_func()
        if self.cap is None:
            return False
        self.running = True
        self.thread  = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()
        return True

    def _reader_loop(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                log.warning("Zone2 RTSP reader: cap closed, reconnecting...")
                self._reconnect()
                continue
            ret, frame = self.cap.read()
            if not ret or frame is None:
                log.warning("Zone2 RTSP reader: read failed, reconnecting...")
                self._reconnect()
                continue
            with self.lock:
                self.latest_frame   = frame
                self.last_read_time = time.time()

    def _reconnect(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        time.sleep(0.5)
        self.cap = self.open_func()

    def read_latest(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass


class CameraZone2:
    def __init__(self, fusion: FusionLayer):
        self.fusion = fusion
        self.model  = YOLO(str(YOLO_WEIGHTS))

        try:
            self.model.fuse()
            log.info("YOLO Zone 2 model fused for faster inference")
        except Exception as e:
            log.warning(f"YOLO Zone 2 fuse skipped: {e}")

        self.device   = 0 if torch.cuda.is_available() else "cpu"
        self.use_half = torch.cuda.is_available()

        log.info(
            f"Zone 2 YOLO device={self.device} half={self.use_half} "
            f"imgsz={YOLO_IMGSZ} reid={ENABLE_ZONE2_REID}"
        )

        self.tracker          = Tracker()
        self._running         = False
        self._last_reid_time  = {}

        self._fps            = 0.0
        self._last_fps_time  = time.time()
        self._frame_count    = 0

    def run(self):
        if CAM_ZONE2_TYPE == "webcam":
            stream = LatestRTSPReader(self._open_webcam)
            stream_label = f"webcam index={CAM_ZONE2_INDEX}"
        elif CAM_ZONE2_TYPE == "rtsp":
            stream = LatestRTSPReader(self._open_rtsp)
            stream_label = "RTSP"
        else:
            log.error(f"Unsupported CAM_ZONE2_TYPE={CAM_ZONE2_TYPE}")
            return

        if not stream.start():
            log.error(f"Zone 2 camera cannot start: {stream_label} open failed")
            return

        log.info(f"Zone 2 camera started (latest-frame reader, {stream_label})")
        self._running = True

        win = "Zone 2 - Packing"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 960, 540)

        while self._running:
            frame = stream.read_latest()
            if frame is None:
                time.sleep(0.02)
                continue

            frame = cv2.resize(frame, (1280, 720))
            annotated = self._process(frame)

            ph, pw = annotated.shape[:2]
            scale   = min(1600 / pw, 900 / ph)
            dw, dh  = int(pw * scale), int(ph * scale)
            display = cv2.resize(annotated, (dw, dh), interpolation=cv2.INTER_AREA)
            cv2.imshow(win, display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self._running = False
        stream.stop()
        try:
            cv2.destroyWindow(win)
        except Exception:
            pass
        log.info("Zone 2 camera stopped")

    def stop(self):
        self._running = False

    def _open_webcam(self):
        cap = cv2.VideoCapture(CAM_ZONE2_INDEX, cv2.CAP_MSMF)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if cap.isOpened():
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            log.info(
                f"Zone 2 camera opened: index={CAM_ZONE2_INDEX} "
                f"resolution={actual_w}x{actual_h}"
            )
            return cap

        log.error(f"Zone 2: cannot open camera index={CAM_ZONE2_INDEX}")
        return None

    # ── RTSP open ─────────────────────────────────────────────────────────────
    def _open_rtsp(self, retries: int = 3):
        for attempt in range(retries):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{RTSP_TRANSPORT}|"
                "fflags;nobuffer|"
                "flags;low_delay|"
                "framedrop;1|"
                "max_delay;100000|"
                "probesize;32|"
                "analyzeduration;0"
            )
            cap = cv2.VideoCapture(CAM_ZONE2_RTSP, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 15)

            if cap.isOpened():
                log.info(
                    f"RTSP Zone 2 opened transport={RTSP_TRANSPORT} "
                    f"(attempt {attempt+1}/{retries})"
                )
                return cap
            log.warning(f"RTSP Zone 2 open failed attempt {attempt+1}/{retries}")
            time.sleep(0.5)

        log.error("Cannot open Zone 2 RTSP after retries")
        return None

    # ── Core processing ───────────────────────────────────────────────────────
    def _process(self, frame: np.ndarray) -> np.ndarray:
        self._update_fps()

        try:
            results = self.model(
                frame,
                imgsz=YOLO_IMGSZ,
                classes=YOLO_CLASSES,
                conf=YOLO_CONF,
                device=self.device,
                half=self.use_half,
                verbose=False,
            )[0]
        except Exception as e:
            log.warning(f"YOLO inference failed half={self.use_half}: {e}")
            self.use_half = False
            results = self.model(
                frame, imgsz=YOLO_IMGSZ, classes=YOLO_CLASSES,
                conf=YOLO_CONF, device=self.device, half=False, verbose=False,
            )[0]

        dets    = Detections.from_ultralytics(results)
        tracked = self.tracker.update(dets)

        if tracked.tracker_id is None:
            self._draw_overlay(frame)
            return frame

        now = time.time()
        h, w = frame.shape[:2]

        for i, track_id in enumerate(tracked.tracker_id):
            track_id = int(track_id)
            x1, y1, x2, y2 = tracked.xyxy[i].astype(int)
            x1 = max(0, min(x1, w-1))
            x2 = max(0, min(x2, w-1))
            y1 = max(0, min(y1, h-1))
            y2 = max(0, min(y2, h-1))

            if x2 <= x1 or y2 <= y1:
                continue

            # ── Re-ID ────────────────────────────────────────────────────────
            # Luôn chạy theo interval kể cả khi đã confirmed
            # để face có cơ hội override lock sai
            if ENABLE_ZONE2_REID:
                last_time   = self._last_reid_time.get(track_id, 0.0)
                should_reid = (now - last_time >= REID_INTERVAL_SEC)

                if should_reid:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        try:
                            self.fusion.identify_zone2(track_id, crop, frame_bgr=frame)
                        except Exception as e:
                            log.warning(f"Zone2 identify failed track={track_id}: {e}")
                    self._last_reid_time[track_id] = now

            # ── Label + color ─────────────────────────────────────────────────
            if ENABLE_ZONE2_REID:
                confirmed = self.fusion.is_confirmed(track_id, zone="zone2")
                label     = self.fusion.get_label(track_id, zone="zone2")
                color     = COLOR_CONFIRMED if confirmed else COLOR_UNKNOWN
            else:
                label = f"track_{track_id}"
                color = COLOR_UNKNOWN

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(20, y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        self._draw_overlay(frame)
        return frame

    def _update_fps(self):
        self._frame_count += 1
        now     = time.time()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self._fps         = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_time = now

    def _draw_overlay(self, frame: np.ndarray):
        reid_text = "ON" if ENABLE_ZONE2_REID else "OFF"
        cv2.putText(frame, "ZONE 2", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(
            frame,
            f"SRC:{CAM_ZONE2_TYPE.upper()} IMG:{YOLO_IMGSZ} ReID:{reid_text}",
            (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2,
        )
        if SHOW_FPS:
            cv2.putText(frame, f"FPS:{self._fps:.1f}", (8, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
