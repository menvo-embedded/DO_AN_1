# Run Demo Guide

This guide describes the recommended demo flow for the active root-level
project. Use `python_cv/main.py` as the PC entry point.

## 1. Prepare Python Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If CUDA is required, install the correct PyTorch build first, then install the
remaining requirements.

## 2. Prepare Runtime Config

```powershell
Copy-Item .env.example python_cv\.env
```

Edit `python_cv/.env` only on the demo machine.

AI dataset and raw video folders default to the internal drive:

```text
WAREHOUSE_DATASET_ROOT=D:/warehouse_dataset
DATASET_CROPS_ROOT=D:/warehouse_dataset/dataset_crops
RAW_VIDEO_ROOT=D:/warehouse_dataset/raw_videos_full
REVIEW_CROP_ROOT=D:/warehouse_dataset/review_crops_reid_yoloseg
IMOU_SD_VIDEO_ROOT=D:/warehouse_dataset/raw_videos_from_imou_sd
```

The demo does not require the old external `F:` drive. If a machine uses a
different data location, edit these values in `python_cv/.env`.

To copy local data from the old `F:` drive to `D:` without deleting `F:`:

```powershell
.\scripts\migrate_warehouse_dataset_F_to_D.ps1
.\scripts\verify_dataset_paths.ps1
```

For a simple CV-only demo:

```text
ENABLE_MQTT=false
ENABLE_ZONE1=false
ENABLE_ZONE2=true
CAM_ZONE2_TYPE=webcam
CAM_ZONE2_INDEX=0
ENABLE_INSIGHTFACE=true
```

For RFID + CV end-to-end:

```text
ENABLE_MQTT=true
ENABLE_ZONE1=true
ENABLE_ZONE2=true
MQTT_BROKER=<broker-ip>
MQTT_TOPIC=warehouse/rfid/scan
```

## 3. Prepare Models and Galleries

Expected local files:

- `python_cv/yolo11n-seg.pt` or root `yolo11n-seg.pt`
- `python_cv/models/reid_resnet50_v3_cleanval.pth`
- `python_cv/data/gallery/gallery.pkl`
- Optional: `python_cv/data/face_gallery/insightface_gallery.pkl`

Do not submit private galleries or raw face/person datasets unless they are
anonymized and approved.

If another machine does not have the full dataset under `D:\warehouse_dataset`,
the runtime demo can still work when the required YOLO/Re-ID model files and
`python_cv/data/gallery/gallery.pkl` are present. In that case, gallery preload
from dataset crops will be skipped because the gallery already exists.

## 4. Start MQTT Broker for End-to-End Demo

Start a local or LAN MQTT broker, then confirm the broker IP matches:

- ESP32 firmware `MQTT_BROKER`
- `python_cv/.env` `MQTT_BROKER`

The ESP32 publishes JSON payloads to:

```json
{
  "uid": "CARD_UID",
  "employee_id": "NV001",
  "timestamp": "2026-05-07T10:00:00+07:00",
  "device": "door-01",
  "zone": 1
}
```

## 5. Flash ESP32 Firmware

From the repository root:

```powershell
platformio run -d firmware -t upload
platformio device monitor -d firmware
```

Before final submission, move WiFi/MQTT secrets out of
`firmware/src/main.cpp` and into an untracked local config file.

## 6. Run PC Pipeline

```powershell
cd python_cv
python main.py
```

Expected windows/services:

- Zone 2 camera window if `ENABLE_ZONE2=true`
- Zone 1 door camera window if `ENABLE_ZONE1=true`
- Flask dashboard at `http://localhost:5000`
- MQTT listener if `ENABLE_MQTT=true`

Press `q` in an OpenCV camera window or `Ctrl+C` in the terminal to stop.

## 7. Demo Scenarios

- CV-only: person appears in Zone 2, system detects/tracks and identifies when
  face/body confidence is sufficient.
- RFID + Zone 1: scan a valid RFID card, cross the door line, and verify entry
  log is created.
- Anomaly: scan an unknown UID or cross Zone 1 without RFID and verify anomaly
  log appears.
- Dashboard: verify entries, RFID events, anomalies, and realtime presence.

## 8. Known Demo Limits

- Zone 3 is optional and disabled by default. Enable `ENABLE_ZONE3=true` only
  after configuring `CAM_ZONE3_TYPE` and the matching camera source.
- Current demo defaults are tuned for Zone 2 webcam and InsightFace.
- Raw videos, debug images, and gallery pickle files may contain personal data
  and should not be submitted as-is.
