import os
import re
import cv2
import time


def _load_cameras():
    """Load RTSP test targets from env, never from committed passwords."""
    raw = os.getenv("RTSP_TEST_TARGETS", "")
    cameras = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        try:
            ip, password = item.split(",", 1)
        except ValueError:
            print(f"Skip invalid RTSP_TEST_TARGETS item: {item}")
            continue
        cameras.append((ip.strip(), password.strip()))
    return cameras


CAMERAS = _load_cameras()

URL_PATTERNS = [
    "rtsp://admin:{password}@{ip}:554/cam/realmonitor?channel=1&subtype=0",
    "rtsp://admin:{password}@{ip}:554/cam/realmonitor?channel=1&subtype=1",
]

def test_url(name, url):
    print("\n" + "=" * 80)
    print(f"TEST: {name}")
    print(re.sub(r"://([^:]+):([^@]+)@", r"://\1:********@", url))

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("RESULT: FAIL - cannot open RTSP")
        cap.release()
        return False

    ok_frames = 0

    for i in range(30):
        ret, frame = cap.read()

        if ret and frame is not None:
            ok_frames += 1
            h, w = frame.shape[:2]
            print(f"RESULT: OK - frame {w}x{h}")
            cv2.imshow("RTSP TEST", frame)
            cv2.waitKey(500)
            break

        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

    if ok_frames == 0:
        print("RESULT: FAIL - opened but no frame")
        return False

    return True

def main():
    if not CAMERAS:
        print("No RTSP test targets configured.")
        print("Set RTSP_TEST_TARGETS like: 192.168.1.10,password1;192.168.1.11,password2")
        return

    found = []

    for ip, password in CAMERAS:
        for pattern in URL_PATTERNS:
            subtype = "subtype=1" if "subtype=1" in pattern else "subtype=0"
            url = pattern.format(ip=ip, password=password)
            name = f"{ip} | password={password} | {subtype}"

            ok = test_url(name, url)

            if ok:
                found.append((ip, password, subtype, url))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if not found:
        print("Không có tổ hợp nào mở được.")
    else:
        for ip, password, subtype, url in found:
            print(f"OK: ip={ip} | password={password} | {subtype}")
            print(url)

if __name__ == "__main__":
    main()
