# Cleanup Audit Report

Audit date: 2026-05-07  
Project: Warehouse Access RFID + Computer Vision

## 1. Tong Quan Repo Hien Tai

- Repo root `D:\warehouse-access-rfid-cv` la ban dang lam viec chinh.
- Thu muc `warehouse-access-rfid-cv/` ben trong root la ban nested/stale, co code cu hon va khong phai entrypoint hien tai.
- Cac module chinh da co: `firmware/`, `python_cv/`, `hardware/`, `docs/`, `experiments/`.
- Repo hien chua san sang nop nguyen trang vi co secret/local config, raw video, anh nguoi that, gallery pickle, model nang, output debug, cache, file backup va ban nested cu.
- Root khong phai git repo trong moi truong hien tai, nen can kiem tra lai repo/VCS truoc khi dong goi nop.

## 2. Pipeline Chinh Hien Tai

Entrypoint PC-side hien tai: `python_cv/main.py`.

Luang chay hien tai:

- Load config tu `python_cv/config/settings.py`, doc them `python_cv/.env` neu co.
- Khoi tao `Database`, `ReIDEngine`, `Gallery`, optional `InsightFaceEngine`.
- Optional MQTT listener khi `ENABLE_MQTT=true`.
- Chay Zone 2 camera thread khi `ENABLE_ZONE2=true`.
- Chay Flask dashboard thread.
- Chay Zone 1 camera tren main thread khi `ENABLE_ZONE1=true`; neu tat Zone 1 thi service giu alive loop.

File duoc import/chay trong pipeline chinh:

- `python_cv/main.py`
- `python_cv/config/settings.py`
- `python_cv/database/database.py`
- `python_cv/database/models.py`
- `python_cv/dashboard/app.py`
- `python_cv/detection/camera_zone1.py`
- `python_cv/detection/camera_zone2.py`
- `python_cv/fusion/fusion_layer.py`
- `python_cv/mqtt/mqtt_listener.py`
- `python_cv/reid/reid_engine.py`
- `python_cv/reid/gallery.py`
- `python_cv/reid/face_insightface_engine.py`
- `python_cv/tracking/tracker.py`
- `python_cv/utils/logger.py`

Khong nam trong pipeline chinh hien tai:

- `python_cv/detection/camera_zone3.py`: con code nhung current `settings.py` khong define `CAM_ZONE3_RTSP`, va `main.py` khong import/chay Zone 3.
- `python_cv/app.py`: dashboard copy cu, khong duoc import boi `main.py`; ban dung la `python_cv/dashboard/app.py`.
- `python_cv/main_backup_before_*.py`, `*_backup_before_*.py`: backup cu.
- `warehouse-access-rfid-cv/`: ban nested/stale.

## 3. Can Giu De Nop

Nen giu trong ban nop sau khi sanitize:

- `README.md`
- `requirements.txt`
- `.env.example`
- `docs/RUN_DEMO.md`
- `docs/diagrams/`, `docs/proposal/`, `docs/report_assets/` neu bo sung noi dung report.
- `firmware/platformio.ini`
- `firmware/src/main.cpp` sau khi tach WiFi/MQTT secret ra config rieng hoac thay placeholder.
- `firmware/config/device_config.example.h`
- `python_cv/main.py`
- `python_cv/config/settings.py`
- `python_cv/config/thresholds.yaml` neu con dung lam tai lieu threshold; neu khong thi dua vao archive.
- `python_cv/database/`
- `python_cv/dashboard/`
- `python_cv/detection/camera_zone1.py`
- `python_cv/detection/camera_zone2.py`
- `python_cv/fusion/fusion_layer.py`
- `python_cv/mqtt/mqtt_listener.py`
- `python_cv/reid/reid_engine.py`
- `python_cv/reid/gallery.py`
- `python_cv/reid/face_insightface_engine.py`
- `python_cv/reid/trainer.py` neu muon nop phan train model.
- `python_cv/tracking/`
- `python_cv/utils/`
- Mot so tools can thiet: `build_face_gallery_from_dataset_crops.py`, `calibrate_reid_threshold_margin.py`, `capture_zone2_reid_debug.py`, `offline_crop_reid_yoloseg.py`, `standalone_webcam_hybrid_debug.py`.
- `hardware/pcb/warehouse_rfid_controller/*.kicad_*`, `hardware/pcb/gerber/`, `hardware/pcb/warehouse_rfid_controller/custom_footprints.pretty/`, `3d_models/` neu giao vien can kiem tra PCB.

## 4. Nen Archive

Nen move vao `archive/` thay vi xoa ngay:

- `warehouse-access-rfid-cv/` nested stale repo.
- `experiments/old_tests/`.
- `python_cv/main_backup_before_disable_mqtt.py`
- `python_cv/main_backup_before_zone3.py`
- `python_cv/config/settings_backup_before_*.py`
- `python_cv/detection/*_backup_before_*.py`
- `python_cv/fusion/fusion_layer_backup_before_core_fix.py`
- `python_cv/reid/*_backup_before_*.py`
- `python_cv/reid/gallery_backup_before_top5_fix.py`
- `python_cv/app.py` neu dashboard chinh da xac nhan la `python_cv/dashboard/app.py`.
- `python_cv/fix_db.py` sau khi xac nhan DB migration trong `database.py` du.
- `python_cv/code_review_*.txt`
- `python_cv/project_tree.txt`
- `python_cv/gallery_py_current.txt`
- `python_cv/test_image.py`, `python_cv/check_dataset.py` neu khong can trong demo.
- Cac test camera/RTSP cu dung `CAM_ZONE3_RTSP` neu chua cap nhat: `check_3_camera_quality_fps.py`, `record_dataset.py`, `live_crop.py`, `preview_monitor_zone23.py`, `record_raw_scene_imou*.py`, `record_raw_imou_2k.py`.
- Duplicate tool `python_cv/tools/crop_reid_from_video.py.py`.
- `hardware/pcb/warehouse_rfid_controller/.history/`.
- Cac file `FINAL.*`, `warehouse.*`, `warehouse_rfid_controller.*` can chon 1 bo final ro rang; bo con lai nen archive sau khi xac minh voi KiCad.

## 5. Co The Xoa Sau Khi Review

Chi xoa sau khi da backup/confirm:

- `.pio/`
- `firmware/.pio/`
- `.pytest_cache/`
- `.vscode/` neu khong nop cau hinh IDE.
- Tat ca `__pycache__/` va `*.pyc`.
- `python_cv/outputs/` debug frames, logs, review images, demo crops, temp DB; neu can evidence thi copy mot it anh/log da an danh sang `demo_assets/`.
- `python_cv/outputs/warehouse_backup_before_db_fix.db`
- Anh test tai root `python_cv/face_test_*.jpg`, `python_cv/test_crop_*.jpg`, `python_cv/test_face*.jpg`, `python_cv/test.jpg` neu la anh nguoi that va khong can demo.
- Duplicate YOLO weights o root va `python_cv/`; nen chi giu 1 vi tri hoac huong dan download.

## 6. Khong Nen Nop Vi Nhay Cam/Nang

Tuyet doi khong nop nguyen trang:

- `python_cv/.env`: local runtime config co the chua RTSP/MQTT secret.
- `firmware/src/main.cpp` hien co WiFi/MQTT hardcoded; can sanitize truoc khi nop.
- Raw video nguoi that: `python_cv/data/dataset_raw/*.mp4`.
- Person crops/face images nguoi that: `python_cv/data/dataset_crops/`, `python_cv/data/gallery/*/*.jpg`, `python_cv/data/face_gallery/raw/`, `python_cv/outputs/**/crops`, `python_cv/outputs/**/frames`.
- Gallery pickle chua embedding sinh trac hoc: `python_cv/data/gallery/*.pkl`, `python_cv/data/face_gallery/*.pkl`, `python_cv/data/gallery.pkl`.
- Local SQLite co log thuc te: `python_cv/outputs/warehouse.db`.
- Model nang neu khong bat buoc nop: `python_cv/models/*.pth`, `python_cv/models/*.zip`, `python_cv/models/insightface/**`, `python_cv/yolo11*.pt`, root `yolo11*.pt`.

Neu giao vien can chay demo offline, nen nop model qua link rieng hoac `demo_assets/models/README.md` ghi noi tai model, thay vi dua het vao source zip.

## 7. Loi Consistency Can Sua Truoc Khi Nop

- UID mapping firmware va Python khong dong bo:
  - Firmware co NV001-NV007.
  - Python `UID_MAP` hien chi phuc vu NV001-NV005.
  - Mot so UID trong firmware map sang nhan vien khac so voi Python. Can tao 1 bang mapping chuan duy nhat.
- Zone config khong dong bo:
  - Current `main.py` chi chay Zone 1/Zone 2.
  - `camera_zone3.py` con ton tai nhung `settings.py` khong co `CAM_ZONE3_RTSP`.
  - Khong co `ENABLE_ZONE3` trong config hien tai.
- Demo mode chua ro:
  - `.env.example` da duoc lam sach va ghi CV-only default.
  - Can quyet dinh ban nop la CV-only demo hay RFID+CV end-to-end.
- Firmware config:
  - `firmware/config/device_config.example.h` ton tai nhung `firmware/src/main.cpp` chua include/su dung.
  - Can tach WiFi SSID/password, MQTT host/topic/client va UID mapping ra config local/untracked hoac thay placeholder.
- Dataset/model path:
  - `main.py` preload gallery tu `DATASET_CROPS_ROOT`; default moi la `D:/warehouse_dataset/dataset_crops`.
  - Neu `gallery.pkl` da co thi khong preload lai; nhung ban nop can huong dan ro cach lay/build gallery.
- Encoding:
  - Mot so README/docs cu bi mojibake tieng Viet, can rewrite UTF-8 sach neu dua vao report.
- Test/scripts:
  - Nhieu script/test import `CAM_ZONE3_RTSP` nen se fail voi current settings.
  - Mot so test la placeholder, khong co gia tri cham diem.

## 8. Dependency Thieu Trong requirements.txt

Da bo sung root `requirements.txt` cho pipeline chinh:

- `python-dotenv`
- `scipy`
- `insightface`
- `onnxruntime`
- `Pillow`
- `git+https://github.com/KaiyangZhou/deep-person-reid.git`

Luu y:

- Neu dung GPU cho InsightFace, co the can `onnxruntime-gpu` thay `onnxruntime`.
- Cac test cu dung `face_recognition`, nhung pipeline chinh khong dung. Chi them neu quyet dinh giu test face_recognition cu.
- PyTorch nen cai theo command chinh thuc phu hop CUDA/may demo.

## 9. Cac Buoc Chay Demo De Xuat

Demo CV-only ngan gon:

1. `python -m venv .venv`
2. `.\.venv\Scripts\Activate.ps1`
3. `pip install -r requirements.txt`
4. `Copy-Item .env.example python_cv\.env`
5. Trong `python_cv/.env`: `ENABLE_MQTT=false`, `ENABLE_ZONE1=false`, `ENABLE_ZONE2=true`, `CAM_ZONE2_TYPE=webcam`.
6. `cd python_cv`
7. `python main.py`
8. Mo `http://localhost:5000`.

Demo RFID + CV end-to-end:

1. Sanitize va flash firmware ESP32.
2. Start MQTT broker.
3. Dong bo `MQTT_BROKER`, `MQTT_TOPIC`, `MQTT_CLIENT` giua firmware va `python_cv/.env`.
4. Dong bo UID mapping firmware va Python.
5. Set `ENABLE_MQTT=true`, `ENABLE_ZONE1=true`, `ENABLE_ZONE2=true`.
6. Chay `python_cv/main.py`.
7. Quet the, di qua line Zone 1, xem entries/anomalies/presence tren dashboard.

## 10. Ket Luan Audit

Repo da co prototype ky thuat tot va du cac subsystem can thiet, nhung chua san
sang nop nguyen trang. Can clean secret/data/output, dong bo mapping/config,
chot pipeline demo, va tach ro phan source code voi model/dataset/evidence.
