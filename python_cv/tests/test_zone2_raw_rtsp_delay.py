import os
import cv2
import time

from config.settings import CAM_ZONE2_RTSP

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;udp|"
    "fflags;nobuffer|"
    "flags;low_delay|"
    "framedrop;1|"
    "max_delay;100000|"
    "probesize;32|"
    "analyzeduration;0"
)

cap = cv2.VideoCapture(CAM_ZONE2_RTSP, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 15)

if not cap.isOpened():
    print("Cannot open RTSP")
    raise SystemExit

print("RTSP opened. Press q to quit.")

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Read failed")
        time.sleep(0.2)
        continue

    cv2.putText(
        frame,
        f"RAW RTSP TEST {time.strftime('%H:%M:%S')}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Zone2 RAW RTSP Delay Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()