import os
from dotenv import load_dotenv

import cv2
import numpy as np

from alerts.ntfy_alert import NtfyAlert


load_dotenv()

frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

cv2.putText(
    frame,
    "TEST NTFY ALERT",
    (70, 230),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.2,
    (0, 0, 255),
    3,
)

cv2.putText(
    frame,
    "Warehouse RFID + CV",
    (95, 290),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (0, 0, 0),
    2,
)

alert = NtfyAlert()

print("ENABLE_NTFY_ALERT =", os.getenv("ENABLE_NTFY_ALERT"))
print("NTFY_TOPIC =", os.getenv("NTFY_TOPIC"))

alert.send_text("TEST: Gui thong bao text tu Python ve topic kho_hang")
alert.last_sent_time = 0
alert.send_frame(
    frame,
    message="TEST: Gui anh canh bao tu he thong RFID-CV ve ntfy",
    prefix="test_ntfy",
)
