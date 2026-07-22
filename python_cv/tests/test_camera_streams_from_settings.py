import sys
from pathlib import Path

# Add python_cv root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
from config.settings import CAMERA_STREAMS


def mask_rtsp(url):
    text = str(url)
    if "rtsp://admin:" in text and "@" in text:
        prefix = "rtsp://admin:"
        start = text.find(prefix) + len(prefix)
        end = text.find("@", start)
        return text[:start] + "********" + text[end:]
    return text


def main():
    print("Camera streams from settings:")
    print("=" * 70)

    for zone, cfg in CAMERA_STREAMS.items():
        if not cfg.get("enabled", False):
            print(f"{zone}: disabled")
            continue

        print("\n" + "=" * 70)
        print(f"TEST {zone}: {cfg.get('name')}")
        print("type  :", cfg.get("type"))
        print("source:", mask_rtsp(cfg.get("source")))

        source = cfg.get("source")

        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print("RESULT: FAIL - cannot open")
            cap.release()
            continue

        ok = False

        for _ in range(30):
            ret, frame = cap.read()

            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"RESULT: OK - frame {w}x{h}")
                ok = True
                break

        if not ok:
            print("RESULT: FAIL - opened but no frame")

        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
