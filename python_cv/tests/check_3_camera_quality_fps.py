# tests/check_3_camera_quality_fps.py
# Check độ phân giải thật + FPS thật của Zone 1 / Zone 2 / Zone 3

import sys
import time
import re
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import CAM_ZONE1_INDEX, CAM_ZONE2_RTSP, CAM_ZONE3_RTSP


OUTPUT_DIR = ROOT_DIR / "outputs" / "debug_frames" / "camera_quality_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_SECONDS = 10

# Zone 1 test nhiều mức phân giải
ZONE1_TEST_RESOLUTIONS = [
    (640, 480),
    (1280, 720),
    (1920, 1080),
]

# Với RTSP: subtype=1 thường là stream nhẹ, subtype=0 thường là main stream 1080p
TEST_RTSP_SUBTYPES = [1, 0]


def mask_rtsp(url: str) -> str:
    if not url:
        return ""
    return re.sub(r"://([^:/]+):([^@]+)@", r"://\1:***@", url)


def make_rtsp_subtype(url: str, subtype: int) -> str:
    if not url:
        return url

    if "subtype=" in url:
        return re.sub(r"subtype=\d+", f"subtype={subtype}", url)

    if "?" in url:
        return url + f"&subtype={subtype}"

    return url + f"?subtype={subtype}"


def read_test(cap, name, seconds=10, save_sample=True):
    start = time.time()
    frames = 0
    failed = 0
    first_shape = None
    last_frame = None

    while time.time() - start < seconds:
        ret, frame = cap.read()

        if not ret or frame is None:
            failed += 1
            time.sleep(0.01)
            continue

        frames += 1
        last_frame = frame

        if first_shape is None:
            first_shape = frame.shape

    elapsed = time.time() - start
    actual_fps = frames / elapsed if elapsed > 0 else 0

    prop_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    prop_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    prop_fps = cap.get(cv2.CAP_PROP_FPS)

    if first_shape is not None:
        h, w = first_shape[:2]
    else:
        h, w = 0, 0

    print("\n==========", name, "==========")
    print("Opened          :", cap.isOpened())
    print("Frame shape     :", f"{w}x{h}")
    print("CAP_PROP size   :", f"{int(prop_w)}x{int(prop_h)}")
    print("CAP_PROP_FPS    :", prop_fps)
    print("Actual FPS      :", round(actual_fps, 2))
    print("Frames read     :", frames)
    print("Read failed     :", failed)

    if save_sample and last_frame is not None:
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", name)
        path = OUTPUT_DIR / f"{safe_name}_{w}x{h}_{round(actual_fps, 1)}fps.jpg"
        cv2.imwrite(str(path), last_frame)
        print("Sample saved    :", path)

    return {
        "name": name,
        "opened": cap.isOpened(),
        "width": w,
        "height": h,
        "cap_fps": prop_fps,
        "actual_fps": actual_fps,
        "frames": frames,
        "failed": failed,
    }


def test_zone1():
    print("\n\n############################")
    print("TEST ZONE 1 - WEBCAM/DROIDCAM")
    print("############################")
    print("CAM_ZONE1_INDEX =", CAM_ZONE1_INDEX)

    results = []

    for w, h in ZONE1_TEST_RESOLUTIONS:
        cap = cv2.VideoCapture(CAM_ZONE1_INDEX, cv2.CAP_DSHOW)

        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(CAM_ZONE1_INDEX)

        # Ưu tiên MJPG để webcam có thể lên FPS/1080 tốt hơn
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print(f"[FAIL] Zone1 cannot open at {w}x{h}")
            cap.release()
            continue

        result = read_test(cap, f"zone1_index{CAM_ZONE1_INDEX}_{w}x{h}", TEST_SECONDS)
        results.append(result)
        cap.release()
        time.sleep(0.5)

    return results


def test_rtsp_zone(zone_name, base_url):
    print("\n\n############################")
    print(f"TEST {zone_name} - RTSP")
    print("############################")

    results = []

    if not base_url:
        print(f"[SKIP] {zone_name} RTSP empty")
        return results

    for subtype in TEST_RTSP_SUBTYPES:
        url = make_rtsp_subtype(base_url, subtype)

        print(f"\n[INFO] Opening {zone_name} subtype={subtype}")
        print("[INFO]", mask_rtsp(url))

        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            print("[WARN] CAP_FFMPEG failed, retry default backend...")
            cap.release()
            cap = cv2.VideoCapture(url)

        if not cap.isOpened():
            print(f"[FAIL] Cannot open {zone_name} subtype={subtype}")
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        result = read_test(cap, f"{zone_name}_subtype{subtype}", TEST_SECONDS)
        results.append(result)

        cap.release()
        time.sleep(0.5)

    return results


def print_summary(all_results):
    print("\n\n================ SUMMARY ================")

    for r in all_results:
        print(
            f"{r['name']:<28} | "
            f"opened={r['opened']} | "
            f"{r['width']}x{r['height']} | "
            f"actual_fps={r['actual_fps']:.2f} | "
            f"cap_fps={r['cap_fps']}"
        )

    print("\nSample frames saved at:")
    print(OUTPUT_DIR)


def main():
    all_results = []

    all_results.extend(test_zone1())
    all_results.extend(test_rtsp_zone("zone2", CAM_ZONE2_RTSP))
    all_results.extend(test_rtsp_zone("zone3", CAM_ZONE3_RTSP))

    print_summary(all_results)


if __name__ == "__main__":
    main()