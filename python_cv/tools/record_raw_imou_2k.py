# tools/record_raw_imou_2k.py
# Record raw video 2K từ Imou Zone 2 / Zone 3, giữ nguyên chất lượng stream subtype=0

import sys
import time
import re
import argparse
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import CAM_ZONE2_RTSP, CAM_ZONE3_RTSP, RAW_VIDEO_2K_ROOT


OUTPUT_ROOT = RAW_VIDEO_2K_ROOT

FORCE_SUBTYPE = 0
WINDOW_NAME = "RAW IMOU 2K RECORDER"

ROTATE_MODE = None
MAX_READ_FAILS = 60


def mask_rtsp(url: str) -> str:
    return re.sub(r"://([^:/]+):([^@]+)@", r"://\1:***@", url)


def force_subtype(url: str, subtype: int) -> str:
    if "subtype=" in url:
        return re.sub(r"subtype=\d+", f"subtype={subtype}", url)
    if "?" in url:
        return url + f"&subtype={subtype}"
    return url + f"?subtype={subtype}"


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


def open_camera(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        raise RuntimeError("Không mở được camera RTSP")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", required=True, choices=["zone2", "zone3"])
    parser.add_argument("--id", required=True, help="VD: NV001 / NV002 / NV005 / mixed")
    parser.add_argument("--name", default="", help='VD: "Bo Man"')
    parser.add_argument("--scenario", default="clean", help="clean / walk / sit / hard / mixed")
    parser.add_argument("--seconds", type=int, default=60)
    args = parser.parse_args()

    zone = args.zone
    emp_id = args.id.strip()
    emp_name = args.name.strip().replace(" ", "_")
    scenario = args.scenario.strip()
    seconds = int(args.seconds)

    base_rtsp = CAM_ZONE2_RTSP if zone == "zone2" else CAM_ZONE3_RTSP
    rtsp_url = force_subtype(base_rtsp, FORCE_SUBTYPE)

    print("[INFO] Zone:", zone)
    print("[INFO] RTSP:", mask_rtsp(rtsp_url))
    print("[INFO] Recording seconds:", seconds)

    cap = open_camera(rtsp_url)

    # Đọc frame đầu để lấy đúng độ phân giải thật
    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError("Không đọc được frame đầu tiên")

    frame = rotate_frame(frame)
    h, w = frame.shape[:2]

    fps_prop = cap.get(cv2.CAP_PROP_FPS)
    fps_save = fps_prop if fps_prop and fps_prop > 0 else 25.0

    out_dir = OUTPUT_ROOT / zone / emp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_name = emp_name if emp_name else emp_id

    video_path = out_dir / f"{zone}_{emp_id}_{safe_name}_{scenario}_{w}x{h}_{ts}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps_save, (w, h))

    if not writer.isOpened():
        raise RuntimeError("Không tạo được VideoWriter")

    print("========== RECORD START ==========")
    print("Save:", video_path)
    print("Resolution:", f"{w}x{h}")
    print("FPS save:", fps_save)
    print("Q/ESC: stop early")

    start = time.time()
    frame_count = 0
    read_fails = 0
    fps_live = 0.0
    prev = time.time()

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            read_fails += 1
            print(f"[WARN] Read fail {read_fails}/{MAX_READ_FAILS}")

            if read_fails >= MAX_READ_FAILS:
                print("[WARN] Too many read fails, stop.")
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

        preview = cv2.resize(frame, (960, 540))
        cv2.putText(preview, f"REC {zone} | {emp_id} | {scenario}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(preview, f"RAW: {w}x{h} | FPS: {fps_live:.1f} | Left: {remain}s", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(preview, "Q/ESC: stop", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
    print("Video:", video_path)
    print("Frames:", frame_count)
    print("Duration:", round(time.time() - start, 2), "seconds")


if __name__ == "__main__":
    main()
