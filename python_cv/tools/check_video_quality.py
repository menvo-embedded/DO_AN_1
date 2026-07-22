import cv2
import sys
from pathlib import Path
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python tools/check_video_quality.py <video_path>")
    raise SystemExit

video_path = Path(sys.argv[1])

if not video_path.exists():
    print("File not found:", video_path)
    raise SystemExit

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("Cannot open video:", video_path)
    raise SystemExit

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps if fps and fps > 0 else 0

print("=" * 80)
print("VIDEO QUALITY CHECK")
print("=" * 80)
print("File       :", video_path)
print("Resolution :", f"{width}x{height}")
print("FPS        :", round(fps, 2))
print("Frames     :", frame_count)
print("Duration   :", round(duration, 2), "seconds")
print("=" * 80)

sample_count = 20
indices = np.linspace(0, max(frame_count - 1, 0), sample_count).astype(int)

brightness_values = []
blur_values = []

saved_dir = Path("outputs/video_quality_samples") / video_path.stem
saved_dir.mkdir(parents=True, exist_ok=True)

for idx in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ret, frame = cap.read()

    if not ret or frame is None:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    brightness_values.append(brightness)
    blur_values.append(blur_score)

    out_file = saved_dir / f"sample_{idx:06d}_bright_{brightness:.1f}_blur_{blur_score:.1f}.jpg"
    cv2.imwrite(str(out_file), frame)

cap.release()

if brightness_values:
    print("Brightness avg :", round(float(np.mean(brightness_values)), 2))
    print("Brightness min :", round(float(np.min(brightness_values)), 2))
    print("Brightness max :", round(float(np.max(brightness_values)), 2))

if blur_values:
    print("Blur avg       :", round(float(np.mean(blur_values)), 2))
    print("Blur min       :", round(float(np.min(blur_values)), 2))
    print("Blur max       :", round(float(np.max(blur_values)), 2))

print("=" * 80)
print("Sample frames saved to:", saved_dir)
print("=" * 80)

print("QUICK JUDGEMENT:")

if width >= 1920 and height >= 1080:
    print("- Resolution: GOOD for Re-ID crop")
elif width >= 1280 and height >= 720:
    print("- Resolution: ACCEPTABLE")
else:
    print("- Resolution: LOW, crop may be weak")

if fps >= 25:
    print("- FPS: GOOD")
else:
    print("- FPS: LOW")

if blur_values and np.mean(blur_values) >= 100:
    print("- Sharpness: OK")
else:
    print("- Sharpness: may be blurry, check sample frames")

if brightness_values and 50 <= np.mean(brightness_values) <= 210:
    print("- Brightness: OK")
else:
    print("- Brightness: may be too dark/too bright")
