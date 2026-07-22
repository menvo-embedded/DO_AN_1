# python_cv/main.py
"""
Entry point — chạy toàn bộ pipeline:
  Thread 1 (daemon) : MQTT listener
  Thread 2 (daemon) : Camera Zone 2 (RTSP Imou)
  Thread 3 (daemon) : Camera Zone 3 (RTSP)
  Thread 4 (daemon) : Flask dashboard
  Main thread       : Camera Zone 1 (webcam USB — blocking, có GUI)
"""

import sys
import threading
import signal
import random
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings        import FLASK_PORT
from database.database      import Database
from reid.reid_engine       import ReIDEngine
from reid.gallery           import Gallery
from fusion.fusion_layer    import FusionLayer
from mqtt.mqtt_listener     import RFIDListener
from detection.camera_zone1 import CameraZone1
from detection.camera_zone2 import CameraZone2
from detection.camera_zone3 import CameraZone3
from dashboard.app          import run_dashboard
from utils.logger           import get_logger

log = get_logger("main")

DATASET_ROOT   = Path("F:/warehouse_dataset/dataset_crops")
VALID_CLASSES  = ["NV001","NV002","NV003","NV004","NV005"]
GALLERY_PRELOAD = 30  # so anh/nguoi khi preload


def preload_gallery(gallery: Gallery, reid: ReIDEngine):
    """Load san gallery tu dataset truoc khi chay realtime."""
    if gallery.employees():
        log.info(f"Gallery already loaded: {gallery.employees()}")
        return
    total = 0
    for nv in VALID_CLASSES:
        folder = DATASET_ROOT / nv
        if not folder.exists():
            log.warning(f"Dataset folder not found: {folder}")
            continue
        imgs = list(folder.glob("*.jpg"))
        if not imgs:
            continue
        samples = random.sample(imgs, min(GALLERY_PRELOAD, len(imgs)))
        for p in samples:
            img = cv2.imread(str(p))
            if img is None:
                continue
            emb = reid.get_embedding(img)
            if emb is not None:
                gallery.update(nv, emb)
                total += 1
    log.info(f"Gallery pre-loaded: {total} embeddings | {gallery.employees()}")


def main():
    log.info("=" * 50)
    log.info("Warehouse RFID+CV System starting")
    log.info("=" * 50)

    # ── Init core components ─────────────────────────────────────────────────
    db      = Database()
    reid    = ReIDEngine()
    gallery = Gallery()

    # Pre-load gallery tu dataset
    preload_gallery(gallery, reid)

    fusion  = FusionLayer(reid, gallery, db)

    # ── Thread 1: MQTT ───────────────────────────────────────────────────────
    mqtt = RFIDListener(on_event=fusion.on_rfid_event)
    mqtt.start()

    # ── Thread 2: Zone 2 RTSP ────────────────────────────────────────────────
    cam2 = CameraZone2(fusion)
    t_cam2 = threading.Thread(
        target=cam2.run, name="zone2-camera", daemon=True)
    t_cam2.start()

    # ── Thread 3: Zone 3 RTSP ────────────────────────────────────────────────
    cam3 = CameraZone3(fusion)
    t_cam3 = threading.Thread(
        target=cam3.run, name="zone3-camera", daemon=True)
    t_cam3.start()

    # ── Thread 4: Dashboard ──────────────────────────────────────────────────
    t_dash = threading.Thread(
        target=run_dashboard, args=(db,), name="dashboard", daemon=True)
    t_dash.start()
    log.info(f"Dashboard: http://localhost:{FLASK_PORT}")

    # ── Graceful shutdown ────────────────────────────────────────────────────
    def _shutdown(sig, frame):
        log.info("Shutdown signal received")
        cam1.stop()
        cam2.stop()
        cam3.stop()
        mqtt.stop()
        db.close()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Main thread: Zone 1 webcam (blocking) ────────────────────────────────
    cam1 = CameraZone1(fusion)
    cam1.run()  # Nhan 'q' tren cua so Zone 1 de thoat


if __name__ == "__main__":
    main()