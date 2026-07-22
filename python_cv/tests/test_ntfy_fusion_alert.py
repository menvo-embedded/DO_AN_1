import time
from datetime import datetime

import cv2
import numpy as np

from fusion.fusion_layer import FusionLayer


class DummyReID:
    pass


class DummyGallery:
    pass


class DummyDb:
    pass


def make_demo_frame():
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 245
    cv2.rectangle(frame, (180, 80), (460, 400), (0, 0, 255), 3)
    cv2.putText(
        frame,
        "FUSION ANOMALY TEST",
        (70, 225),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        3,
    )
    cv2.putText(
        frame,
        "Warehouse RFID + CV",
        (125, 285),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
    )
    return frame


if __name__ == "__main__":
    fusion = FusionLayer(DummyReID(), DummyGallery(), DummyDb())
    frame = make_demo_frame()

    print("Sending ntfy fusion image alert...")
    fusion._send_ntfy_alert(
        "demo_unknown_person",
        "zone2",
        datetime.now(),
        track_id="zone2:test",
        detail="TEST: Fusion Layer alert path with frame",
        frame=frame,
        prefix="test_fusion_alert",
    )

    time.sleep(2)

    if fusion._ntfy_alert is not None:
        fusion._ntfy_alert.last_sent_time = 0

    print("Sending ntfy fusion text alert...")
    fusion._send_ntfy_alert(
        "demo_rfid_no_crossing",
        "zone1",
        datetime.now(),
        employee_id="NV001",
        detail="TEST: Fusion Layer text-only alert path",
        prefix="test_fusion_text",
    )

    time.sleep(5)
    print("Done.")
