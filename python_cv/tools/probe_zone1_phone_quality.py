import cv2
import time
import csv
from pathlib import Path

CAM_INDEX = 3
OUT_CSV = Path("outputs/camera_probe_zone1_phone.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# Các mức cần test
CANDIDATE_RESOLUTIONS = [
    (640, 480),
    (800, 600),
    (960, 540),
    (1280, 720),
    (1600, 900),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
]

CANDIDATE_FPS = [15, 24, 30, 60]

def test_setting(width, height, fps):
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        return {
            "request_width": width,
            "request_height": height,
            "request_fps": fps,
            "opened": False,
            "actual_width": 0,
            "actual_height": 0,
            "actual_fps_prop": 0,
            "measured_fps": 0,
            "read_ok": False,
        }

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    time.sleep(1.0)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps_prop = cap.get(cv2.CAP_PROP_FPS)

    # Đọc vài frame để đo FPS thực tế
    frames = 0
    read_ok = False
    start = time.time()

    for _ in range(90):
        ret, frame = cap.read()
        if ret and frame is not None:
            read_ok = True
            frames += 1

    elapsed = time.time() - start
    measured_fps = frames / elapsed if elapsed > 0 else 0

    cap.release()

    return {
        "request_width": width,
        "request_height": height,
        "request_fps": fps,
        "opened": True,
        "actual_width": actual_w,
        "actual_height": actual_h,
        "actual_fps_prop": round(actual_fps_prop, 2),
        "measured_fps": round(measured_fps, 2),
        "read_ok": read_ok,
    }

results = []

print("=" * 80)
print(f"TEST PHONE CAMERA QUALITY | CAM_INDEX={CAM_INDEX}")
print("=" * 80)

for width, height in CANDIDATE_RESOLUTIONS:
    for fps in CANDIDATE_FPS:
        r = test_setting(width, height, fps)
        results.append(r)

        status = "OK" if r["read_ok"] else "FAIL"
        print(
            f"{status} | request={width}x{height}@{fps} "
            f"-> actual={r['actual_width']}x{r['actual_height']} "
            f"| prop_fps={r['actual_fps_prop']} "
            f"| measured_fps={r['measured_fps']}"
        )

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

print("=" * 80)
print("DONE")
print("CSV saved to:", OUT_CSV)
print("=" * 80)
