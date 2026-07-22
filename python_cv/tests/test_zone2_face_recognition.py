# tests/test_zone2_face_recognition.py
# Test Face Detection / Face Encoding cho Zone 2 IP Camera RTSP
#
# Chạy từ project root:
# cd C:\Users\menvo\Downloads\warehouse-access-rfid-cv\warehouse-access-rfid-cv\python_cv
# D:\UV4\anaconda3\python.exe .\tests\test_zone2_face_recognition.py

import sys
import time
import re
from pathlib import Path

import cv2
import numpy as np
import face_recognition


# ============================================================
# PROJECT ROOT
# ============================================================
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import CAM_ZONE2_RTSP


# ============================================================
# CONFIG - CHỈNH TOÀN BỘ Ở ĐÂY
# ============================================================
ROTATE_MODE = None
# None  : không xoay
# "cw"  : xoay 90 độ theo chiều kim đồng hồ
# "ccw" : xoay 90 độ ngược chiều kim đồng hồ
# "180" : xoay 180 độ

DETECT_MODEL = "hog"
# "hog" : nhẹ hơn, chạy CPU tốt hơn
# "cnn" : mạnh hơn HOG nhưng rất chậm nếu dlib không có CUDA

DETECT_SCALE = 1.5
# 1.0 = giữ nguyên frame, FPS cao hơn
# 1.5 = phóng to frame để bắt mặt nhỏ tốt hơn
# 2.0 = mạnh hơn nhưng chậm hơn

UPSAMPLE = 1
# 0 = nhanh, khó bắt mặt nhỏ
# 1 = cân bằng
# 2 = bắt mặt nhỏ tốt hơn nhưng chậm

AUTO_SAVE_FACE = True
SAVE_COOLDOWN_SEC = 2.0

MIN_FACE_W = 20
MIN_FACE_H = 20
MAX_FACE_RATIO = 2.2

VALIDATE_LANDMARKS = False
# False = dễ detect mặt góc cao hơn
# True  = lọc false positive tốt hơn nhưng có thể loại nhầm mặt thật

OUTPUT_DIR = ROOT_DIR / "outputs" / "debug_frames" / "face_recognition_zone2_test"
WINDOW_NAME = "Zone 2 Face Recognition Test"

MAX_READ_FAILS = 60


# ============================================================
# UTILS
# ============================================================
def mask_rtsp_url(url: str) -> str:
    """Ẩn password RTSP khi in log."""
    if not url:
        return ""

    return re.sub(r"://([^:/]+):([^@]+)@", r"://\1:***@", url)


def rotate_frame(frame):
    """Xoay frame nếu camera bị xoay sai hướng."""
    if ROTATE_MODE is None:
        return frame

    if ROTATE_MODE == "cw":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    if ROTATE_MODE == "ccw":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if ROTATE_MODE == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)

    return frame


def open_zone2_camera():
    """Mở camera Zone 2 từ CAM_ZONE2_RTSP trong settings.py."""
    print("[INFO] Opening Zone 2 RTSP camera...")
    print(f"[INFO] CAM_ZONE2_RTSP = {mask_rtsp_url(CAM_ZONE2_RTSP)}")

    cap = cv2.VideoCapture(CAM_ZONE2_RTSP, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("[WARN] CAP_FFMPEG failed, retry with default backend...")
        cap.release()
        cap = cv2.VideoCapture(CAM_ZONE2_RTSP)

    if not cap.isOpened():
        raise RuntimeError(
            "Không mở được Zone 2 RTSP. Kiểm tra CAM_ZONE2_RTSP trong .env/settings.py, "
            "IP camera, mạng WiFi/LAN hoặc RTSP URL."
        )

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def restore_face_locations(face_locations, scale):
    """
    Face locations lấy trên frame đã resize.
    Cần quy đổi bbox về kích thước frame gốc.
    """
    if scale == 1.0:
        return face_locations

    restored = []

    for top, right, bottom, left in face_locations:
        restored.append(
            (
                int(top / scale),
                int(right / scale),
                int(bottom / scale),
                int(left / scale),
            )
        )

    return restored


def clip_location(loc, frame_w, frame_h):
    """Giới hạn bbox không vượt ra ngoài frame."""
    top, right, bottom, left = loc

    top = max(0, min(top, frame_h - 1))
    bottom = max(0, min(bottom, frame_h - 1))
    left = max(0, min(left, frame_w - 1))
    right = max(0, min(right, frame_w - 1))

    return top, right, bottom, left


def check_face_box(loc):
    """Lọc bbox quá nhỏ hoặc méo bất thường."""
    top, right, bottom, left = loc

    face_w = right - left
    face_h = bottom - top

    if face_w <= 0 or face_h <= 0:
        return False, "invalid"

    if face_w < MIN_FACE_W or face_h < MIN_FACE_H:
        return False, f"small {face_w}x{face_h}"

    ratio = max(face_w / face_h, face_h / face_w)
    if ratio > MAX_FACE_RATIO:
        return False, f"bad_ratio {ratio:.2f}"

    return True, "ok"


def validate_face_landmarks(rgb_frame, loc):
    """
    Lọc false positive bằng landmark.
    Nếu bật VALIDATE_LANDMARKS=True, mặt cần có mắt trái, mắt phải, mũi.
    """
    try:
        landmarks_list = face_recognition.face_landmarks(
            rgb_frame,
            face_locations=[loc],
            model="small",
        )

        if len(landmarks_list) == 0:
            return False, "no_landmark"

        lm = landmarks_list[0]

        has_left_eye = "left_eye" in lm and len(lm["left_eye"]) >= 2
        has_right_eye = "right_eye" in lm and len(lm["right_eye"]) >= 2
        has_nose = "nose_tip" in lm and len(lm["nose_tip"]) >= 2

        if not (has_left_eye and has_right_eye and has_nose):
            return False, "bad_landmark"

        return True, "landmark_ok"

    except Exception:
        return False, "landmark_err"


def save_snapshot(frame, label="manual_full_frame"):
    """Lưu full frame hiện tại khi bấm S."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    ms = int((time.time() % 1) * 1000)

    path = OUTPUT_DIR / f"zone2_{label}_{ts}_{ms:03d}.jpg"
    cv2.imwrite(str(path), frame)

    print(f"[SAVE] Snapshot: {path}")


def save_face_result(frame_draw, frame_raw, loc, encoding):
    """Lưu full bbox, crop mặt và encoding 128D."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    ms = int((time.time() % 1) * 1000)

    prefix = f"zone2_face_{ts}_{ms:03d}"

    top, right, bottom, left = loc
    face_crop = frame_raw[top:bottom, left:right]

    full_path = OUTPUT_DIR / f"{prefix}_full_bbox.jpg"
    crop_path = OUTPUT_DIR / f"{prefix}_crop.jpg"
    enc_path = OUTPUT_DIR / f"{prefix}_encoding.npy"

    cv2.imwrite(str(full_path), frame_draw)
    cv2.imwrite(str(crop_path), face_crop)
    np.save(str(enc_path), encoding)

    print("[SAVE] Face saved:")
    print(f"       {full_path}")
    print(f"       {crop_path}")
    print(f"       {enc_path}")


def draw_overlay(frame, raw_count, valid_count, fps, last_save_text):
    """Hiển thị thông tin debug lên frame."""
    lines = [
        "ZONE 2 FACE DETECTION DIAGNOSE",
        f"Raw faces: {raw_count} | Valid faces: {valid_count}",
        f"Detect model: {DETECT_MODEL} | Upsample: {UPSAMPLE} | Detect scale: {DETECT_SCALE}",
        f"Rotate: {ROTATE_MODE} | Landmark filter: {VALIDATE_LANDMARKS} | FPS: {fps:.1f}",
        "Q/ESC: quit | S: save current full frame",
        "Note: Zone 2 high angle may fail face detection.",
    ]

    if last_save_text:
        lines.append(last_save_text)

    y = 25

    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 24


# ============================================================
# MAIN
# ============================================================
def main():
    if not CAM_ZONE2_RTSP:
        raise RuntimeError("CAM_ZONE2_RTSP đang rỗng. Kiểm tra .env hoặc config/settings.py")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = open_zone2_camera()

    print("[INFO] Zone 2 face recognition test started.")
    print("[INFO] Đứng dưới Zone 2, ngẩng mặt lên camera 1-2 giây.")
    print("[INFO] Q hoặc ESC: thoát.")
    print("[INFO] S: lưu full frame hiện tại.")
    print("[INFO] Nếu Raw faces = 0 liên tục thì chưa detect được mặt.")
    print("[INFO] Nếu có bbox nhưng crop không phải mặt thì đó là false positive.")

    read_fails = 0
    last_save_time = 0.0
    last_save_text = ""

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            read_fails += 1
            print(f"[WARN] Cannot read frame. fail={read_fails}/{MAX_READ_FAILS}")

            if read_fails >= MAX_READ_FAILS:
                print("[WARN] Too many read fails. Reconnecting...")
                cap.release()
                time.sleep(1.0)
                cap = open_zone2_camera()
                read_fails = 0

            time.sleep(0.03)
            continue

        read_fails = 0

        frame = rotate_frame(frame)
        frame_raw = frame.copy()
        frame_draw = frame.copy()

        frame_h, frame_w = frame.shape[:2]

        # FPS
        now = time.time()
        dt = now - prev_time
        prev_time = now

        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # Resize frame để detect mặt nhỏ tốt hơn
        if DETECT_SCALE != 1.0:
            detect_frame = cv2.resize(
                frame,
                None,
                fx=DETECT_SCALE,
                fy=DETECT_SCALE,
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            detect_frame = frame

        rgb_detect = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
        rgb_detect = np.ascontiguousarray(rgb_detect)

        # Detect face
        face_locations_scaled = face_recognition.face_locations(
            rgb_detect,
            number_of_times_to_upsample=UPSAMPLE,
            model=DETECT_MODEL,
        )

        face_locations = restore_face_locations(face_locations_scaled, DETECT_SCALE)

        raw_count = len(face_locations)
        valid_count = 0

        rgb_full = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
        rgb_full = np.ascontiguousarray(rgb_full)

        for loc in face_locations:
            loc = clip_location(loc, frame_w, frame_h)
            top, right, bottom, left = loc

            ok, reason = check_face_box(loc)

            if not ok:
                color = (0, 0, 255)

                cv2.rectangle(frame_draw, (left, top), (right, bottom), color, 2)
                cv2.putText(
                    frame_draw,
                    reason,
                    (left, max(20, top - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

                continue

            if VALIDATE_LANDMARKS:
                lm_ok, lm_reason = validate_face_landmarks(rgb_full, loc)

                if not lm_ok:
                    color = (0, 0, 255)

                    cv2.rectangle(frame_draw, (left, top), (right, bottom), color, 2)
                    cv2.putText(
                        frame_draw,
                        lm_reason,
                        (left, max(20, top - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

                    continue

            # Extract encoding 128D
            encodings = face_recognition.face_encodings(
                rgb_full,
                known_face_locations=[loc],
                num_jitters=1,
                model="small",
            )

            if len(encodings) == 0:
                color = (0, 165, 255)

                cv2.rectangle(frame_draw, (left, top), (right, bottom), color, 2)
                cv2.putText(
                    frame_draw,
                    "face_no_encoding",
                    (left, max(20, top - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

                continue

            encoding = encodings[0]
            valid_count += 1

            color = (0, 255, 0)

            cv2.rectangle(frame_draw, (left, top), (right, bottom), color, 2)
            cv2.putText(
                frame_draw,
                f"face | enc={encoding.shape[0]}d",
                (left, max(20, top - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

            if AUTO_SAVE_FACE:
                current_time = time.time()

                if current_time - last_save_time >= SAVE_COOLDOWN_SEC:
                    save_face_result(frame_draw, frame_raw, loc, encoding)
                    last_save_time = current_time
                    last_save_text = f"Last saved: {time.strftime('%H:%M:%S')}"

        draw_overlay(frame_draw, raw_count, valid_count, fps, last_save_text)

        cv2.imshow(WINDOW_NAME, frame_draw)

        key = cv2.waitKey(1) & 0xFF

        if key in [ord("q"), ord("Q"), 27]:
            print("[INFO] Quit.")
            break

        if key in [ord("s"), ord("S")]:
            save_snapshot(frame_draw, label="manual_full_frame")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()