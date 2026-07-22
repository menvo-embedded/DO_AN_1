import threading
import time
from datetime import datetime

import cv2
import numpy as np
from supervision import Detections
from ultralytics import YOLO

from config.settings import (
    CAM_ZONE1_INDEX,
    CAM_ZONE1_ROTATE,
    YOLO_WEIGHTS,
    YOLO_CONF,
    YOLO_CLASSES,
    ZONE1_ENTRY_LINE_RATIO,
    ZONE1_ENTRY_DIRECTION,
)
from tracking.tracker import Tracker
from fusion.fusion_layer import FusionLayer
from utils.logger import get_logger


log = get_logger("zone1")

COLOR_CONFIRMED = (0, 220, 80)
COLOR_PENDING = (0, 165, 255)
COLOR_LINE = (0, 0, 220)

CROSSING_COOLDOWN = 3.0  # giây

PREVIEW_W = 1600
PREVIEW_H = 900


def apply_zone1_rotation(frame: np.ndarray) -> np.ndarray:
    if CAM_ZONE1_ROTATE in ("none", "0", "off"):
        return frame
    if CAM_ZONE1_ROTATE in ("90cw", "90_cw", "cw", "clockwise"):
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if CAM_ZONE1_ROTATE in ("90ccw", "90_ccw", "ccw", "counterclockwise"):
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if CAM_ZONE1_ROTATE in ("180", "flip"):
        return cv2.rotate(frame, cv2.ROTATE_180)

    log.warning(f"Unknown CAM_ZONE1_ROTATE={CAM_ZONE1_ROTATE}, using raw frame")
    return frame


class CameraZone1:
    def __init__(self, fusion: FusionLayer):
        self.fusion = fusion
        self.model = YOLO(str(YOLO_WEIGHTS))
        self.tracker = Tracker()

        self._prev_cx: dict[int, float] = {}
        self._last_crossing: dict[int, float] = {}
        self._running = False

        # Thread-safe buffer — RFID trigger đọc từ MQTT thread
        self._latest_persons: list = []
        self._persons_lock = threading.Lock()

        # Full annotated frame mới nhất — fusion lấy để lưu ảnh evidence
        self._latest_annotated = None
        self._annotated_lock = threading.Lock()

    def run(self):
        cap = cv2.VideoCapture(CAM_ZONE1_INDEX, cv2.CAP_MSMF)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            log.error(f"Zone 1: cannot open camera index={CAM_ZONE1_INDEX}")
            return

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info(
            f"Zone 1 camera opened: index={CAM_ZONE1_INDEX} "
            f"resolution={actual_w}x{actual_h} rotate={CAM_ZONE1_ROTATE}"
        )
        self._running = True

        win = "Zone 1 - Door"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 960, 540)

        while self._running:
            ret, frame = cap.read()

            if not ret or frame is None:
                log.warning("Zone1: frame read failed - retry")
                time.sleep(0.03)
                continue

            frame = apply_zone1_rotation(frame)
            frame = cv2.resize(frame, (1280, 720))

            annotated = self._process(frame)

            with self._annotated_lock:
                self._latest_annotated = annotated.copy()

            ph, pw = annotated.shape[:2]
            scale   = min(PREVIEW_W / pw, PREVIEW_H / ph)
            dw, dh  = int(pw * scale), int(ph * scale)
            display = cv2.resize(annotated, (dw, dh), interpolation=cv2.INTER_AREA)
            cv2.imshow(win, display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyWindow(win)

    def stop(self):
        self._running = False

    def get_best_person(self, max_age_sec: float = 2.0) -> dict | None:
        """
        Trả về person tốt nhất trong frame hiện tại.
        Score = bbox_area_ratio * 0.6 + confidence * 0.4
        Lọc bỏ detection quá cũ hơn max_age_sec.
        Keys: track_id, crop_bgr, bbox, bbox_area, bbox_area_ratio, confidence, timestamp
        Thread-safe.
        """
        now = time.time()
        with self._persons_lock:
            candidates = [
                p for p in self._latest_persons
                if now - p["timestamp"] <= max_age_sec
            ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda p: p["bbox_area_ratio"] * 0.6 + p["confidence"] * 0.4,
        )

    def get_latest_annotated(self):
        """Trả về full annotated frame Zone 1 mới nhất (đã vẽ bbox + entry line).
        Dùng để lưu ảnh evidence khi quẹt thẻ. Thread-safe. None nếu chưa có frame."""
        with self._annotated_lock:
            if self._latest_annotated is None:
                return None
            return self._latest_annotated.copy()

    def _process(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        entry_line_x = int(w * ZONE1_ENTRY_LINE_RATIO)

        results = self.model(
            frame,
            classes=YOLO_CLASSES,
            conf=YOLO_CONF,
            verbose=False,
        )[0]

        dets = Detections.from_ultralytics(results)
        tracked = self.tracker.update(dets)

        now = datetime.now().astimezone()
        now_ts = time.time()

        track_ids = tracked.tracker_id if tracked.tracker_id is not None else []
        detected  = []

        for i, track_id in enumerate(track_ids):
            x1, y1, x2, y2 = tracked.xyxy[i].astype(int)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            bbox_area       = (x2 - x1) * (y2 - y1)
            bbox_area_ratio = bbox_area / (w * h)
            conf_val        = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            crop            = frame[y1:y2, x1:x2]

            if crop.size > 0:
                detected.append({
                    "track_id":        track_id,
                    "crop_bgr":        crop.copy(),
                    "bbox":            (x1, y1, x2, y2),
                    "bbox_area":       bbox_area,
                    "bbox_area_ratio": bbox_area_ratio,
                    "confidence":      conf_val,
                    "timestamp":       now_ts,
                })

            cx      = float((x1 + x2) / 2)
            prev_cx = self._prev_cx.get(track_id, cx)

            # Quy ước: TRÁI vạch = NGOÀI cửa, PHẢI vạch = TRONG kho.
            # → VÀO KHO = băng trái→phải; RA = băng phải→trái (ZONE1_ENTRY_DIRECTION="lr").
            cross_lr = prev_cx < entry_line_x <= cx   # trái → phải
            cross_rl = prev_cx > entry_line_x >= cx   # phải → trái
            if ZONE1_ENTRY_DIRECTION == "rl":
                is_entry, is_exit = cross_rl, cross_lr
            else:
                is_entry, is_exit = cross_lr, cross_rl

            last_cross_ts = self._last_crossing.get(track_id, 0)

            if (is_entry or is_exit) and (now_ts - last_cross_ts) > CROSSING_COOLDOWN:
                self._last_crossing[track_id] = now_ts
                direction = "ENTRY(vao kho)" if is_entry else "EXIT(ra ngoai)"

                log.info(
                    f"CROSSING {direction}: zone1:track_{track_id} "
                    f"cx={cx:.0f} prev={prev_cx:.0f} line={entry_line_x}"
                )

                # Chỉ hướng VÀO KHO mới kích hoạt fusion (ghép RFID / phát hiện intruder).
                if is_entry and crop.size > 0:
                    self.fusion.on_entry_crossing(
                        "zone1",
                        track_id,
                        crop,
                        now,
                        frame_bgr=frame,
                    )

            self._prev_cx[track_id] = cx

            label = self.fusion.get_label(track_id, zone="zone1")
            confirmed = self.fusion.is_confirmed(track_id, zone="zone1")
            color = COLOR_CONFIRMED if confirmed else COLOR_PENDING

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                f"{label} cx={cx:.0f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        # Cập nhật buffer thread-safe
        with self._persons_lock:
            self._latest_persons = detected

        # Tô màu bán trong suốt 2 vùng (phủ SAU khi đã trích crop → Re-ID/khuôn mặt
        # vẫn dùng ảnh gốc sạch): ngoài cửa = cam, trong kho = xanh lá.
        overlay = frame.copy()
        OUT_COLOR = (0, 140, 255)   # cam  - NGOAI CUA
        IN_COLOR  = (60, 190, 60)   # xanh - TRONG KHO
        if ZONE1_ENTRY_DIRECTION == "rl":
            left_color, right_color = IN_COLOR, OUT_COLOR
        else:
            left_color, right_color = OUT_COLOR, IN_COLOR
        cv2.rectangle(overlay, (0, 0), (entry_line_x, h), left_color, -1)
        cv2.rectangle(overlay, (entry_line_x, 0), (w, h), right_color, -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

        cv2.line(frame, (entry_line_x, 0), (entry_line_x, h), COLOR_LINE, 2)

        cv2.putText(
            frame,
            f"ENTRY LINE x={entry_line_x}",
            (entry_line_x + 5, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_LINE,
            1,
            cv2.LINE_AA,
        )

        # Nhãn phân định 2 phía của vạch (ASCII vì OpenCV không hiển thị dấu tiếng Việt)
        if ZONE1_ENTRY_DIRECTION == "rl":
            left_lbl, right_lbl = "TRONG KHO", "NGOAI CUA"
        else:
            left_lbl, right_lbl = "NGOAI CUA", "TRONG KHO"
        y_side = h // 2
        cv2.putText(frame, f"{left_lbl} <", (max(10, entry_line_x - 240), y_side),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_LINE, 2, cv2.LINE_AA)
        cv2.putText(frame, f"> {right_lbl}", (entry_line_x + 12, y_side),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_LINE, 2, cv2.LINE_AA)

        cv2.putText(
            frame,
            "ZONE 1",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return frame
