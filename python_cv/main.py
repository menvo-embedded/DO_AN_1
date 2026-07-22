# python_cv/main.py
"""
Entry point — chạy pipeline CV-only ưu tiên hiện tại:

  Optional Thread : MQTT listener, chỉ chạy khi ENABLE_MQTT=True
  Thread 1        : Camera Zone 2 (RTSP Imou)
  Thread 2        : Flask dashboard
  Main thread     : Camera Zone 1 (phone webcam index 3 — blocking, có GUI)

MQTT được tạm tắt/bật bằng ENABLE_MQTT trong config/settings.py hoặc .env.
"""

import sys
import threading
import signal
import random
import time
import cv2
from pathlib import Path


ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# IMPORTS
# ============================================================

from config.settings import (
    DATASET_CROPS_ROOT,
    FLASK_PORT,
    ENABLE_MQTT,
    ENABLE_INSIGHTFACE,
    ENABLE_ZONE1,
    ENABLE_ZONE2,
    FACE_DET_SIZE,
    FACE_DET_THRESH,
    FACE_GALLERY_PATH,
    FACE_MODEL_NAME,
    FACE_MATCH_THRESHOLD,
    FACE_MATCH_MARGIN,
)
from database.database import Database
from reid.reid_engine import ReIDEngine
from reid.gallery import Gallery
from fusion.fusion_layer import FusionLayer
from mqtt.mqtt_listener import RFIDListener
from detection.camera_zone1 import CameraZone1
from detection.camera_zone2 import CameraZone2
from dashboard.app import run_dashboard
from utils.logger import get_logger


log = get_logger("main")


# ============================================================
# DATASET / GALLERY PRELOAD CONFIG
# ============================================================

DATASET_ROOT = DATASET_CROPS_ROOT

VALID_CLASS_FOLDERS = {
    "NV001": ["NV001"],
    "NV002": ["NV002"],
    "NV003": ["NV003"],
    "NV004": ["NV004"],
    "NV005": ["NV005", "NV005_city_20260503"],
}

GALLERY_PRELOAD = 30


def preload_gallery(gallery: Gallery, reid: ReIDEngine):
    """
    Load sẵn gallery body Re-ID từ dataset clean trước khi chạy realtime.
    Nếu gallery.pkl đã có data thì không preload lại.
    """
    if gallery.employees():
        log.info(f"Gallery already loaded: {gallery.employees()}")
        return

    total = 0

    for emp_id, folder_names in VALID_CLASS_FOLDERS.items():
        imgs = []

        for folder_name in folder_names:
            folder = DATASET_ROOT / folder_name

            if not folder.exists():
                log.warning(f"Dataset folder not found: {folder}")
                continue

            imgs.extend(folder.glob("*.jpg"))

        if not imgs:
            log.warning(f"No images found for {emp_id}: {folder_names}")
            continue

        samples = random.sample(imgs, min(GALLERY_PRELOAD, len(imgs)))

        for p in samples:
            img = cv2.imread(str(p))

            if img is None:
                continue

            emb = reid.get_embedding(img)

            if emb is not None:
                gallery.update(emp_id, emb)
                total += 1

    log.info(f"Gallery pre-loaded: {total} embeddings | {gallery.employees()}")


def main():
    log.info("=" * 50)
    log.info("Warehouse RFID+CV System starting")
    log.info("=" * 50)

    db = None
    mqtt = None
    cam1 = None
    cam2 = None

    try:
        # ====================================================
        # INIT CORE COMPONENTS
        # ====================================================

        db = Database()
        reid = ReIDEngine()
        gallery = Gallery()
        face_engine = None

        preload_gallery(gallery, reid)

        if ENABLE_INSIGHTFACE:
            from reid.face_insightface_engine import InsightFaceEngine

            log.info(
                f"InsightFace config: FACE_MODEL_NAME={FACE_MODEL_NAME} "
                f"FACE_DET_SIZE={FACE_DET_SIZE} FACE_DET_THRESH={FACE_DET_THRESH}"
            )

            try:
                face_engine = InsightFaceEngine(
                    gallery_path=str(FACE_GALLERY_PATH),
                    model_name=FACE_MODEL_NAME,
                    use_gpu=True,
                    det_size=FACE_DET_SIZE,
                    det_thresh=FACE_DET_THRESH,
                    face_threshold=FACE_MATCH_THRESHOLD,
                    face_margin=FACE_MATCH_MARGIN,
                )
                log.info("InsightFace enabled for Zone 2 hybrid Re-ID")
            except RuntimeError as e:
                face_engine = None
                log.error(f"InsightFace disabled: {e}")
        else:
            log.warning("InsightFace disabled - Zone 2 uses body Re-ID only")

        fusion = FusionLayer(reid, gallery, db, face_engine=face_engine)

        # ====================================================
        # OPTIONAL MQTT THREAD
        # ====================================================

        if ENABLE_MQTT:
            mqtt = RFIDListener(on_event=fusion.on_rfid_event)
            mqtt.start()
            log.info("MQTT listener started")
        else:
            log.warning("MQTT disabled - running CV/Re-ID only")

        # ====================================================
        # CAMERA ZONE 2 THREAD
        # ====================================================

        if ENABLE_ZONE2:
            cam2 = CameraZone2(fusion)
            t_cam2 = threading.Thread(
                target=cam2.run,
                name="zone2-camera",
                daemon=True,
            )
            t_cam2.start()
            log.info("Zone 2 thread started")
        else:
            log.warning("Zone 2 disabled")

        # ====================================================
        # DASHBOARD THREAD
        # ====================================================

        t_dash = threading.Thread(
            target=run_dashboard,
            args=(db,),
            name="dashboard",
            daemon=True,
        )
        t_dash.start()
        log.info(f"Dashboard: http://localhost:{FLASK_PORT}")

        # ====================================================
        # GRACEFUL SHUTDOWN
        # ====================================================

        def _shutdown(sig=None, frame=None):
            log.info("Shutdown signal received")

            if cam1 is not None:
                cam1.stop()

            if cam2 is not None:
                cam2.stop()

            if mqtt is not None:
                mqtt.stop()

            if db is not None:
                db.close()

            sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        # ====================================================
        # CAMERA ZONE 1 MAIN THREAD
        # ====================================================

        if ENABLE_ZONE1:
            cam1 = CameraZone1(fusion)
            fusion.set_zone1(cam1)
            cam1.run()
        else:
            log.warning("Zone 1 disabled - keeping service alive")
            while True:
                time.sleep(1)

        _shutdown()

    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received")

        if cam1 is not None:
            cam1.stop()

        if cam2 is not None:
            cam2.stop()

        if mqtt is not None:
            mqtt.stop()

        if db is not None:
            db.close()

    except Exception as e:
        log.exception(f"Fatal error in main: {e}")

        if cam1 is not None:
            cam1.stop()

        if cam2 is not None:
            cam2.stop()

        if mqtt is not None:
            mqtt.stop()

        if db is not None:
            db.close()

        raise


if __name__ == "__main__":
    main()

