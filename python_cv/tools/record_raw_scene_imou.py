# tools/record_raw_scene_imou.py
# Record RAW video toàn cảnh từ Imou Zone 2 / Zone 3
# Record bằng FFmpeg stream copy: không crop, không detect, không re-encode.
# Đồng thời hiện preview bằng OpenCV để dễ quan sát.
#
# Ví dụ:
# D:\UV4\anaconda3\python.exe .\tools\record_raw_scene_imou.py --zone zone2 --seconds 18000 --session raw_3zone_5h_01

import sys
import re
import time
import shutil
import argparse
import subprocess
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import CAM_ZONE2_RTSP, CAM_ZONE3_RTSP, RAW_VIDEO_ROOT


# ============================================================
# CONFIG
# ============================================================
OUTPUT_ROOT = RAW_VIDEO_ROOT

RECORD_SUBTYPE = 0      # subtype=0: main stream, max quality
PREVIEW_SUBTYPE = 1     # subtype=1: nhẹ hơn để xem preview

WINDOW_W = 960
WINDOW_H = 540

WINDOW_NAME = "RAW IMOU RECORDER PREVIEW"

MAX_READ_FAILS = 120


def force_subtype(url: str, subtype: int) -> str:
    if "subtype=" in url:
        return re.sub(r"subtype=\d+", f"subtype={subtype}", url)
    if "?" in url:
        return url + f"&subtype={subtype}"
    return url + f"?subtype={subtype}"


def mask_rtsp(url: str) -> str:
    return re.sub(r"://([^:/]+):([^@]+)@", r"://\1:***@", url)


def find_ffmpeg():
    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is not None:
        return ffmpeg

    candidates = [
        Path(r"D:\UV4\anaconda3\Library\bin\ffmpeg.exe"),
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
    ]

    winget_root = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    if winget_root.exists():
        candidates.extend(list(winget_root.rglob("ffmpeg.exe")))

    for p in candidates:
        if p.exists():
            return str(p)

    raise RuntimeError("Không tìm thấy ffmpeg.exe")


def open_preview(preview_url):
    cap = cv2.VideoCapture(preview_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(preview_url)

    if not cap.isOpened():
        raise RuntimeError("Không mở được preview RTSP")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def stop_ffmpeg(process):
    if process is None:
        return

    if process.poll() is not None:
        return

    try:
        process.stdin.write(b"q\n")
        process.stdin.flush()
        time.sleep(1.0)
    except Exception:
        pass

    if process.poll() is None:
        try:
            process.terminate()
            time.sleep(1.0)
        except Exception:
            pass

    if process.poll() is None:
        try:
            process.kill()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", required=True, choices=["zone2", "zone3"])
    parser.add_argument("--seconds", type=int, default=18000)
    parser.add_argument("--session", default="raw_session")
    args = parser.parse_args()

    zone = args.zone
    seconds = int(args.seconds)
    session = args.session.strip().replace(" ", "_")

    base_url = CAM_ZONE2_RTSP if zone == "zone2" else CAM_ZONE3_RTSP

    record_url = force_subtype(base_url, RECORD_SUBTYPE)
    preview_url = force_subtype(base_url, PREVIEW_SUBTYPE)

    ts = time.strftime("%Y%m%d_%H%M%S")

    out_dir = OUTPUT_ROOT / session
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{zone}_raw_mainstream_{ts}_{seconds}s.mkv"

    ffmpeg = find_ffmpeg()

    cmd = [
        ffmpeg,
        "-y",
        "-fflags", "+genpts",
        "-rtsp_transport", "tcp",
        "-i", record_url,
        "-t", str(seconds),
        "-map", "0:v:0",
        "-c:v", "copy",
        "-an",
        str(out_path),
    ]

    safe_cmd = " ".join(cmd).replace(record_url, mask_rtsp(record_url))

    print("========== RAW RECORD WITH PREVIEW ==========")
    print("Zone          :", zone)
    print("Session       :", session)
    print("Seconds       :", seconds)
    print("Record RTSP   :", mask_rtsp(record_url))
    print("Preview RTSP  :", mask_rtsp(preview_url))
    print("Output        :", out_path)
    print("Mode          : FFmpeg stream copy, no crop, no detect, no re-encode")
    print("Preview       : OpenCV window")
    print("Q/ESC         : stop record")
    print("============================================")
    print("[CMD]", safe_cmd)

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    cap = open_preview(preview_url)

    start = time.time()
    frame_count = 0
    read_fails = 0
    fps_live = 0.0
    prev = time.time()

    try:
        while True:
            if process.poll() is not None:
                print("[INFO] FFmpeg process finished.")
                break

            ret, frame = cap.read()

            if not ret or frame is None:
                read_fails += 1
                print(f"[WARN] Preview read fail {read_fails}/{MAX_READ_FAILS}")

                if read_fails >= MAX_READ_FAILS:
                    print("[WARN] Preview lost too many frames. Record may still be running.")
                    read_fails = 0

                time.sleep(0.03)
                continue

            read_fails = 0
            frame_count += 1

            now = time.time()
            dt = now - prev
            prev = now

            if dt > 0:
                fps_live = 0.9 * fps_live + 0.1 * (1.0 / dt)

            elapsed = now - start
            remain = max(0, seconds - int(elapsed))

            preview = cv2.resize(frame, (WINDOW_W, WINDOW_H))

            cv2.putText(
                preview,
                f"REC RAW {zone.upper()} | {session}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                preview,
                f"Record: subtype={RECORD_SUBTYPE} max quality | Preview: subtype={PREVIEW_SUBTYPE}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                preview,
                f"Preview FPS: {fps_live:.1f} | Left: {remain}s | Q/ESC: stop",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(f"{WINDOW_NAME} - {zone}", preview)

            key = cv2.waitKey(1) & 0xFF

            if key in [ord("q"), ord("Q"), 27]:
                print("[INFO] Stop by user.")
                stop_ffmpeg(process)
                break

            if elapsed >= seconds:
                print("[INFO] Time reached.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        stop_ffmpeg(process)

    print("========== DONE ==========")
    print("Video saved:", out_path)


if __name__ == "__main__":
    main()
