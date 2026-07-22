# tools/preview_monitor_zone23.py
# Preview theo dõi Zone 2 + Zone 3 khi đang record raw bằng FFmpeg stable.
# Không record, không crop, không detect. Chỉ xem hình + hiện thời gian + dung lượng file mới nhất.

import sys
import re
import time
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import CAM_ZONE2_RTSP, CAM_ZONE3_RTSP, RAW_VIDEO_ROOT


SESSION = "raw_3zone_5h_final_01"
VIDEO_DIR = RAW_VIDEO_ROOT / SESSION

PREVIEW_SUBTYPE = 1
WINDOW_NAME = "ZONE 2 + ZONE 3 RECORD MONITOR"

DISPLAY_W = 640
DISPLAY_H = 360


def force_subtype(url: str, subtype: int) -> str:
    if "subtype=" in url:
        return re.sub(r"subtype=\d+", f"subtype={subtype}", url)
    if "?" in url:
        return url + f"&subtype={subtype}"
    return url + f"?subtype={subtype}"


def open_cam(url):
    cap = cv2.VideoCapture(force_subtype(url, PREVIEW_SUBTYPE), cv2.CAP_FFMPEG)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(force_subtype(url, PREVIEW_SUBTYPE))

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def latest_file_size_mb(prefix):
    if not VIDEO_DIR.exists():
        return 0.0, "no folder"

    files = list(VIDEO_DIR.glob(f"{prefix}*.mkv"))
    if not files:
        return 0.0, "no file"

    latest = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    size_mb = latest.stat().st_size / (1024 * 1024)

    return size_mb, latest.name


def draw_panel(frame, title, elapsed_sec, size_mb, filename):
    frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))

    cv2.putText(frame, title, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, f"Preview subtype={PREVIEW_SUBTYPE} | Record main stream 2K", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.putText(frame, f"Elapsed: {int(elapsed_sec)}s | File: {size_mb:.1f} MB", (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.putText(frame, filename[:70], (15, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    cv2.putText(frame, "Q/ESC: close monitor only", (15, DISPLAY_H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    return frame


def main():
    print("[INFO] Opening preview Zone 2 + Zone 3")
    print("[INFO] Monitoring folder:", VIDEO_DIR)

    cap2 = open_cam(CAM_ZONE2_RTSP)
    cap3 = open_cam(CAM_ZONE3_RTSP)

    if not cap2.isOpened():
        print("[WARN] Zone 2 preview cannot open")
    if not cap3.isOpened():
        print("[WARN] Zone 3 preview cannot open")

    start = time.time()

    while True:
        elapsed = time.time() - start

        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()

        if not ret2 or frame2 is None:
            frame2 = 255 * cv2.UMat(DISPLAY_H, DISPLAY_W, cv2.CV_8UC3).get()
            cv2.putText(frame2, "ZONE 2 NO SIGNAL", (40, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if not ret3 or frame3 is None:
            frame3 = 255 * cv2.UMat(DISPLAY_H, DISPLAY_W, cv2.CV_8UC3).get()
            cv2.putText(frame3, "ZONE 3 NO SIGNAL", (40, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        z2_size, z2_file = latest_file_size_mb("zone2")
        z3_size, z3_file = latest_file_size_mb("zone3")

        panel2 = draw_panel(frame2, "ZONE 2 RECORD MONITOR", elapsed, z2_size, z2_file)
        panel3 = draw_panel(frame3, "ZONE 3 RECORD MONITOR", elapsed, z3_size, z3_file)

        combined = cv2.vconcat([panel2, panel3])

        cv2.imshow(WINDOW_NAME, combined)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord("q"), ord("Q"), 27]:
            break

    cap2.release()
    cap3.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
