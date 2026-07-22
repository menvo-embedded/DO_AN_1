# Clean Source Checklist

Use this checklist before submitting or sharing the project source.

## Keep In Source

- `README.md`, `requirements.txt`, `.env.example`
- `python_cv/main.py`
- `python_cv/config/`
- `python_cv/dashboard/`
- `python_cv/database/`
- `python_cv/detection/`
- `python_cv/fusion/`
- `python_cv/mqtt/`
- `python_cv/reid/`
- `python_cv/tracking/`
- `python_cv/utils/`
- `firmware/src/main.cpp`
- `firmware/config/device_config.example.h`
- `firmware/platformio.ini`
- `hardware/pcb/warehouse_rfid_controller/`
- `hardware/pcb/gerber/`
- `docs/RUN_DEMO.md`

## Exclude From Submitted Source

- `python_cv/.env`
- `python_cv/.env.*`
- `firmware/config/device_config.h`
- `.venv/`, `.pio/`, `firmware/.pio/`, `.pytest_cache/`, `.vscode/`
- `__pycache__/`
- `python_cv/outputs/`
- `python_cv/data/dataset_raw/`
- `python_cv/data/dataset_crops/`
- `python_cv/data/face_gallery/raw/`
- `python_cv/data/**/*.pkl`
- real face/person images and raw videos
- heavy model files: `*.pt`, `*.pth`, `*.onnx`, `*.zip`
- nested stale copy: `warehouse-access-rfid-cv/`
- exported old copy: `_EXPORT_FOR_TEAM_AI/`
- backup/debug files: `*_backup_before_*.py`, `project_tree.txt`, `code_review_*.txt`

## Local Config Notes

Create `python_cv/.env` from `.env.example` only on the demo machine.
Create `firmware/config/device_config.h` from `device_config.example.h` only on the flash machine.
Do not put WiFi passwords, MQTT broker credentials, RTSP passwords, galleries, or biometric data in the submitted source.
