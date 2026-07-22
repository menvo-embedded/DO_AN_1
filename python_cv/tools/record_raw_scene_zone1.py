# tools/record_raw_scene_zone1.py
# Record raw full-frame Zone 1 webcam/DroidCam
# Không crop, không detect, không phân loại.

import sys
import time
import argparse
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import CAM_ZONE1_INDEX, RAW_VIDEO_ROOT


OUTPUT_ROOT = RAW_VIDEO_ROOT

WINDOW_NAME = "RAW ZONE 1 RECORDER"

ROTATE_MODE = "cw"
# Zone 1 trước đó test face đúng đang dùng "cw"
# Nếu muốn không xoay thì đổi thành None

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_SAVE = 30

FOURCC = "MJPG"
# MJPG + AVI giữ chất lượng tốt hơn mp4v, file sẽ hơi nặng

MAX_READ_FAILS = 60


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


def open_zone1_camera():
    print("[INFO] Opening Zone 1 camera...")
    print("[INFO] CAM_ZONE1_INDEX =", CAM_ZONE1_INDEX)

    cap = cv2.VideoCapture(CAM_ZONE1_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(CAM_ZONE1_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Không mở được Zone 1 camera.")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS_SAVE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--session", default="raw_test_zone1")
    args = parser.parse_args()

    seconds = int(args.seconds)
    session = args.session.strip().replace(" ", "_")

    out_dir = OUTPUT_ROOT / session
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")

    video_path = out_dir / f"zone1_raw_webcam_{FRAME_WIDTH}x{FRAME_HEIGHT}_{ts}_{seconds}s.avi"

    cap = open_zone1_camera()

    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError("Không đọc được frame đầu tiên từ Zone 1.")

    frame = rotate_frame(frame)

    h, w = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*FOURCC)
    writer = cv2.VideoWriter(str(video_path), fourcc, FPS_SAVE, (w, h))

    if not writer.isOpened():
        raise RuntimeError("Không tạo được VideoWriter.")

    print("========== RAW ZONE 1 RECORD ==========")
    print("Session :", session)
    print("Seconds :", seconds)
    print("Output  :", video_path)
    print("Size    :", f"{w}x{h}")
    print("FPS     :", FPS_SAVE)
    print("Mode    : full-frame webcam record, no crop, no detect")
    print("Q/ESC   : stop early")
    print("=======================================")

    start = time.time()
    frame_count = 0
    read_fails = 0

    prev = time.time()
    fps_live = 0.0

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            read_fails += 1
            print(f"[WARN] Read fail {read_fails}/{MAX_READ_FAILS}")

            if read_fails >= MAX_READ_FAILS:
                print("[WARN] Too many read fails. Stop.")
                break

            time.sleep(0.02)
            continue

        read_fails = 0

        frame = rotate_frame(frame)

        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))

        writer.write(frame)
        frame_count += 1

        now = time.time()
        dt = now - prev
        prev = now

        if dt > 0:
            fps_live = 0.9 * fps_live + 0.1 * (1.0 / dt)

        elapsed = now - start
        remain = max(0, seconds - int(elapsed))

        preview = frame.copy()
        cv2.putText(
            preview,
            "REC ZONE 1 RAW",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            f"{w}x{h} | FPS: {fps_live:.1f} | Left: {remain}s | Rotate: {ROTATE_MODE}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            "Q/ESC: stop",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, preview)

        key = cv2.waitKey(1) & 0xFF

        if key in [ord("q"), ord("Q"), 27]:
            print("[INFO] Stop by user.")
            break

        if elapsed >= seconds:
            print("[INFO] Done recording.")
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()

    print("========== RECORD DONE ==========")
    print("Video :", video_path)
    print("Frames:", frame_count)
    print("Time  :", round(time.time() - start, 2), "seconds")


if __name__ == "__main__":
    main()
