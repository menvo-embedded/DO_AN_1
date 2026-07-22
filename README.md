# Warehouse Access RFID + Computer Vision

Graduation project prototype for warehouse access control and employee
monitoring using ESP32 RFID, cameras, YOLO detection, tracking, Re-ID,
optional face recognition, fusion logic, SQLite, and a Flask dashboard.

## Main Components

- `firmware/`: ESP32 Arduino firmware for RC522 RFID, LCD I2C, buzzer, servo
  gate, WiFi, NTP, and MQTT event publishing.
- `python_cv/`: main PC-side application for camera processing, Re-ID,
  InsightFace hybrid recognition, RFID/CV fusion, SQLite storage, and dashboard.
- `hardware/`: KiCad PCB source files, custom footprints, 3D models, and Gerber
  export for the RFID controller board.
- `docs/`: submission notes, demo guide, diagrams, and report assets.
- `experiments/`, `python_cv/tools/`, `python_cv/tests/`: debug, calibration,
  data collection, and validation utilities.

The active PC pipeline is `python_cv/main.py`. The nested folder
`warehouse-access-rfid-cv/` is an older/stale copy and should not be used for
the final submission without review.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If CUDA-specific PyTorch wheels are required, install PyTorch first using the
official command for the target GPU/CUDA version, then install the remaining
requirements.

Copy the clean runtime example and edit local values:

```powershell
Copy-Item .env.example python_cv\.env
```

Do not commit `python_cv/.env`, RTSP passwords, WiFi passwords, raw videos,
face/person galleries, or private datasets.

## Dataset and Local Data Path

AI datasets, raw videos, review crops, galleries, and model artifacts are local
runtime assets. They are intentionally not part of the source-code submission.

Default local dataset root:

```text
D:\warehouse_dataset
```

Default subfolders used by tools:

```text
D:\warehouse_dataset\dataset_crops
D:\warehouse_dataset\raw_videos_full
D:\warehouse_dataset\review_crops_reid_yoloseg
D:\warehouse_dataset\raw_videos_from_imou_sd
```

These defaults are configured through `.env`:

```text
WAREHOUSE_DATASET_ROOT=D:/warehouse_dataset
DATASET_CROPS_ROOT=D:/warehouse_dataset/dataset_crops
RAW_VIDEO_ROOT=D:/warehouse_dataset/raw_videos_full
REVIEW_CROP_ROOT=D:/warehouse_dataset/review_crops_reid_yoloseg
IMOU_SD_VIDEO_ROOT=D:/warehouse_dataset/raw_videos_from_imou_sd
```

If another machine uses a different drive or folder, edit `python_cv/.env`
instead of changing code. To copy local data from the old external `F:` drive to
`D:`, review and run:

```powershell
.\scripts\migrate_warehouse_dataset_F_to_D.ps1
.\scripts\verify_dataset_paths.ps1
```

The migration script copies data only; it does not delete anything from `F:`.

## Run Demo

From the repository root:

```powershell
cd python_cv
python main.py
```

Default demo mode can run CV-only with Zone 2 webcam and InsightFace enabled.
For RFID + CV end-to-end, set `ENABLE_MQTT=true`, configure the MQTT broker,
flash the ESP32 firmware, and start the broker before running `python main.py`.
Create `firmware/config/device_config.h` from
`firmware/config/device_config.example.h` before flashing the ESP32.

Dashboard:

```text
http://localhost:5000
```

Detailed demo steps are in `docs/RUN_DEMO.md`.

## Submission Notes

Before submitting, review:

- `CLEANUP_AUDIT_REPORT.md`
- `SUBMISSION_STRUCTURE_PROPOSAL.md`
- `docs/CLEAN_SOURCE_CHECKLIST.md`
- `CLEANUP_COMMANDS_DRAFT.ps1`

The current repo contains local debug outputs, model files, gallery pickles, raw
videos, and real credentials in firmware. Clean or archive them before handing
the project to a reviewer.
