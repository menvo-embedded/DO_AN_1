# tests/test_zone2_insightface.py
# Test Face Detection + Face Embedding bằng InsightFace/SCRFD cho Zone 2
#
# Chạy từ project root:
# cd C:\Users\menvo\Downloads\warehouse-access-rfid-cv\warehouse-access-rfid-cv\python_cv
# D:\UV4\anaconda3\python.exe .\tests\test_zone2_insightface.py

import sys
import time
import re
from pathlib import Path

import cv2
import numpy as np

from insightface.app import FaceAnalysis


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

MODEL_NAME = "buffalo_sc"
# buffalo_sc: nhẹ, phù hợp máy yếu/trung bình
# buffalo_l : nặng hơn, có thể chính xác hơn nếu máy chịu được

USE_GPU = True
# True  : ưu tiên GPU nếu onnxruntime-gpu hoạt động
# False : dùng CPU

DET_SIZE = (640, 640)
# (640, 640)  : cân bằng
# (960, 960)  : bắt mặt nhỏ tốt hơn nhưng chậm hơn
# (1280,1280) : mạnh hơn nữa nhưng rất chậm

DET_THRESH = 0.50
# 0.50 = chặt hơn, ít false positive
# 0.30 = dễ bắt mặt hơn
# 0.20 = rất dễ bắt nhưng dễ false positive

DETECT_EVERY_N_FRAMES = 1
# 1 = detect mỗi frame, chính xác hơn nhưng chậm hơn
# 2/3 = mượt hơn, nhưng bbox cập nhật chậm hơn

AUTO_SAVE_FACE = True
SAVE_COOLDOWN_SEC = 2.0

MIN_FACE_W = 25
MIN_FACE_H = 25
MAX_FACE_RATIO = 2.2

OUTPUT_DIR = ROOT_DIR / "outputs" / "debug_frames" / "insightface_zone2_test"
WINDOW_NAME = "Zone 2 InsightFace Test"

MAX_READ_FAILS = 60


# ============================================================
# UTILS
# ============================================================
def mask_rtsp_url(url: str) -> str:
    if not url:
        return ""

    return re.sub(r"://([^:/]+):([^@]+)@", r"://\1:***@", url)


def rotate_frame(frame):
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


def init_insightface():
    print("[INFO] Loading InsightFace...")
    print(f"[INFO] MODEL_NAME = {MODEL_NAME}")
    print(f"[INFO] USE_GPU = {USE_GPU}")
    print(f"[INFO] DET_SIZE = {DET_SIZE}")
    print(f"[INFO] DET_THRESH = {DET_THRESH}")

    # Một số version insightface cũ không hỗ trợ providers=
    # Nên dùng cách tương thích rộng hơn.
    app = FaceAnalysis(
        name=MODEL_NAME,
        allowed_modules=["detection", "recognition"],
    )

    # ctx_id = 0: ưu tiên GPU nếu môi trường hỗ trợ
    # ctx_id = -1: CPU
    ctx_id = 0 if USE_GPU else -1

    try:
        app.prepare(
            ctx_id=ctx_id,
            det_size=DET_SIZE,
            det_thresh=DET_THRESH,
        )
    except TypeError:
        # Một số version cũ không có det_thresh trong prepare()
        print("[WARN] This InsightFace version may not support det_thresh in prepare(). Retry without det_thresh...")
        app.prepare(
            ctx_id=ctx_id,
            det_size=DET_SIZE,
        )

    print("[INFO] InsightFace ready.")
    return app


def check_face_box(bbox):
    x1, y1, x2, y2 = bbox

    face_w = x2 - x1
    face_h = y2 - y1

    if face_w <= 0 or face_h <= 0:
        return False, "invalid"

    if face_w < MIN_FACE_W or face_h < MIN_FACE_H:
        return False, f"small {int(face_w)}x{int(face_h)}"

    ratio = max(face_w / face_h, face_h / face_w)
    if ratio > MAX_FACE_RATIO:
        return False, f"bad_ratio {ratio:.2f}"

    return True, "ok"


def clip_bbox(bbox, frame_w, frame_h):
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(int(x1), frame_w - 1))
    y1 = max(0, min(int(y1), frame_h - 1))
    x2 = max(0, min(int(x2), frame_w - 1))
    y2 = max(0, min(int(y2), frame_h - 1))

    return x1, y1, x2, y2


def save_snapshot(frame, label="manual_full_frame"):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    ms = int((time.time() % 1) * 1000)

    path = OUTPUT_DIR / f"zone2_{label}_{ts}_{ms:03d}.jpg"
    cv2.imwrite(str(path), frame)

    print(f"[SAVE] Snapshot: {path}")


def save_face_result(frame_draw, frame_raw, bbox, embedding, score):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    ms = int((time.time() % 1) * 1000)

    prefix = f"zone2_insightface_{ts}_{ms:03d}"

    x1, y1, x2, y2 = bbox
    face_crop = frame_raw[y1:y2, x1:x2]

    full_path = OUTPUT_DIR / f"{prefix}_full_bbox.jpg"
    crop_path = OUTPUT_DIR / f"{prefix}_crop.jpg"
    emb_path = OUTPUT_DIR / f"{prefix}_embedding.npy"

    cv2.imwrite(str(full_path), frame_draw)
    cv2.imwrite(str(crop_path), face_crop)
    np.save(str(emb_path), embedding)

    print("[SAVE] InsightFace face saved:")
    print(f"       score={score:.3f}")
    print(f"       embedding_shape={embedding.shape}")
    print(f"       {full_path}")
    print(f"       {crop_path}")
    print(f"       {emb_path}")


def draw_overlay(frame, raw_count, valid_count, fps, last_save_text):
    lines = [
        "ZONE 2 INSIGHTFACE / SCRFD TEST",
        f"Raw faces: {raw_count} | Valid faces: {valid_count}",
        f"Model: {MODEL_NAME} | Det size: {DET_SIZE} | Det thresh: {DET_THRESH}",
        f"Rotate: {ROTATE_MODE} | GPU: {USE_GPU} | FPS: {fps:.1f}",
        "Q/ESC: quit | S: save current full frame",
        "Note: Zone 2 high angle may still fail sometimes.",
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

    app = init_insightface()
    cap = open_zone2_camera()

    print("[INFO] Zone 2 InsightFace test started.")
    print("[INFO] Đứng dưới Zone 2, ngẩng mặt lên camera 1-2 giây.")
    print("[INFO] Q hoặc ESC: thoát.")
    print("[INFO] S: lưu full frame hiện tại.")

    read_fails = 0
    frame_idx = 0

    last_faces = []
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
        frame_idx += 1

        frame = rotate_frame(frame)
        frame_raw = frame.copy()
        frame_draw = frame.copy()

        frame_h, frame_w = frame.shape[:2]

        now = time.time()
        dt = now - prev_time
        prev_time = now

        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        if frame_idx % DETECT_EVERY_N_FRAMES == 0:
            try:
                last_faces = app.get(frame_raw)
            except Exception as e:
                print(f"[WARN] InsightFace get() error: {e}")
                last_faces = []

        raw_count = len(last_faces)
        valid_count = 0

        for face in last_faces:
            bbox = face.bbox
            score = float(face.det_score)

            x1, y1, x2, y2 = clip_bbox(bbox, frame_w, frame_h)
            bbox_clipped = (x1, y1, x2, y2)

            ok, reason = check_face_box(bbox_clipped)

            if not ok:
                color = (0, 0, 255)

                cv2.rectangle(frame_draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame_draw,
                    reason,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

                continue

            embedding = getattr(face, "embedding", None)

            if embedding is None:
                color = (0, 165, 255)

                cv2.rectangle(frame_draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame_draw,
                    f"face score={score:.2f} no_emb",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

                continue

            embedding = np.asarray(embedding, dtype=np.float32)
            valid_count += 1

            color = (0, 255, 0)

            cv2.rectangle(frame_draw, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame_draw,
                f"face score={score:.2f} emb={embedding.shape[0]}d",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

            # Vẽ keypoints nếu có
            kps = getattr(face, "kps", None)
            if kps is not None:
                for px, py in kps.astype(int):
                    cv2.circle(frame_draw, (int(px), int(py)), 2, (0, 255, 255), -1)

            if AUTO_SAVE_FACE:
                current_time = time.time()

                if current_time - last_save_time >= SAVE_COOLDOWN_SEC:
                    save_face_result(frame_draw, frame_raw, bbox_clipped, embedding, score)
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