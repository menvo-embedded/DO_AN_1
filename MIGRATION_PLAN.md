# Migration Plan: C Drive Paths To D Drive

Project root: `D:\warehouse-access-rfid-cv`

This report lists remaining hard-coded `C:` / `C:\Users\menvo` style paths found in text files. No files were deleted or moved during this audit.

## Scan Scope

Command pattern used:

```powershell
rg -n -S "C:\\|C:/|C:\\Users\\menvo|C:/Users/menvo" .
```

Ignored noisy generated folders during review: `__pycache__`, `.pytest_cache`, `.pio`, `python_cv/outputs/logs`, `python_cv/project_tree.txt`, `ENV_TRACE_REPORT.csv`.

Some `C:` matches from UI text such as `Q/ESC: stop` are false positives and are not migration risks.

## Hard-Coded Paths Found

| File | Current path | Used for | Recommended D path / fix | Priority |
|---|---|---|---|---|
| `python_cv/tools/record_raw_scene_imou.py` | `C:\ffmpeg\bin\ffmpeg.exe` | Fallback location for ffmpeg binary used by raw Imou recording helper. | Prefer env var `FFMPEG_EXE`; fallback to `D:\tools\ffmpeg\bin\ffmpeg.exe` or auto-discover `ffmpeg` from `PATH`. | Medium |
| `_EXPORT_FOR_TEAM_AI/python_cv/tools/record_raw_scene_imou.py` | `C:\ffmpeg\bin\ffmpeg.exe` | Duplicate/export copy of the same helper. | Same as above, or regenerate export after fixing source. | Low |
| `python_cv/tests/test_zone2_insightface_identify.py` | `C:\Users\menvo\Downloads\warehouse-access-rfid-cv\warehouse-access-rfid-cv\python_cv` | Comment-only run instruction. | Replace comment with `cd D:\warehouse-access-rfid-cv\python_cv`. | Low |
| `python_cv/tests/test_zone2_insightface.py` | `C:\Users\menvo\Downloads\warehouse-access-rfid-cv\warehouse-access-rfid-cv\python_cv` | Comment-only run instruction. | Replace comment with `cd D:\warehouse-access-rfid-cv\python_cv`. | Low |
| `python_cv/tests/test_zone2_face_recognition.py` | `C:\Users\menvo\Downloads\warehouse-access-rfid-cv\warehouse-access-rfid-cv\python_cv` | Comment-only run instruction. | Replace comment with `cd D:\warehouse-access-rfid-cv\python_cv`. | Low |
| `_EXPORT_FOR_TEAM_AI/python_cv/tests/test_zone2_insightface_identify.py` | `C:\Users\menvo\Downloads\warehouse-access-rfid-cv\warehouse-access-rfid-cv\python_cv` | Export duplicate comment. | Regenerate export after source comment cleanup. | Low |
| `_EXPORT_FOR_TEAM_AI/python_cv/tests/test_zone2_insightface.py` | `C:\Users\menvo\Downloads\warehouse-access-rfid-cv\warehouse-access-rfid-cv\python_cv` | Export duplicate comment. | Regenerate export after source comment cleanup. | Low |
| `_EXPORT_FOR_TEAM_AI/python_cv/tests/test_zone2_face_recognition.py` | `C:\Users\menvo\Downloads\warehouse-access-rfid-cv\warehouse-access-rfid-cv\python_cv` | Export duplicate comment. | Regenerate export after source comment cleanup. | Low |
| `hardware/pcb/warehouse_rfid_controller/warehouse_rfid_controller.kicad_pro` | `C:/Users/menvo/Desktop/` | KiCad plot output directory metadata. | Use project-local output such as `D:\warehouse-access-rfid-cv\hardware\pcb\warehouse_rfid_controller\plots`. | Medium |
| `hardware/pcb/warehouse_rfid_controller/warehouse_rfid_controller.kicad_pcb` | `C:/Users/menvo/Desktop/` | KiCad board output directory metadata. | Use project-local output such as `D:\warehouse-access-rfid-cv\hardware\pcb\warehouse_rfid_controller\plots`. | Medium |
| `hardware/pcb/warehouse_rfid_controller/3d_models/Active_Buzzer.step` | `C:/Users/toby_/Desktop/Toby/3D Part Libraries/...` | STEP file contains original CAD source path/comment metadata. | Usually safe to leave; if KiCad uses an external model path elsewhere, store model under `hardware/pcb/warehouse_rfid_controller/3d_models`. | Low |

## Runtime Cache Paths Not Hard-Coded In Repo

These are not necessarily visible as repo literals, but libraries commonly default to `C:\Users\<user>`:

| Component | Typical C path | Risk | D target proposed |
|---|---|---|---|
| PyTorch / TorchReID weights | `%USERPROFILE%\.cache\torch` | Downloads/checkpoints can grow large on C. | `D:\warehouse_runtime\torch` via `TORCH_HOME` |
| pip cache | `%LOCALAPPDATA%\pip\Cache` | Package cache can grow large. | `D:\warehouse_runtime\pip-cache` via `PIP_CACHE_DIR` |
| temp files | `%TEMP%`, `%TMP%` | Model/video/temp extraction can hit C. | `D:\warehouse_runtime\temp` via `TEMP` and `TMP` |
| Ultralytics config/cache | `%APPDATA%\Ultralytics` or user config dir | YOLO settings can write under user profile. | `D:\warehouse_runtime\ultralytics` via `YOLO_CONFIG_DIR` and `ULTRALYTICS_CONFIG_DIR` |
| Conda package cache | Anaconda package cache under C/user env | Package cache can grow large. | `D:\warehouse_runtime\conda-pkgs` via `CONDA_PKGS_DIRS` |
| InsightFace model cache | `%USERPROFILE%\.insightface` | Buffalo model downloads can land on C. | Junction `%USERPROFILE%\.insightface` -> `D:\warehouse_runtime\.insightface` |

## Recommended Migration Actions

1. Run `migrate_to_d_drive.ps1` once from PowerShell.
2. Restart terminal/IDE so user-level environment variables are reloaded.
3. Run `verify_after_migration.ps1` to test imports, CUDA, `.env`, cameras, and MQTT port.
4. Later cleanup, with review only: update `record_raw_scene_imou.py` to use `FFMPEG_EXE` or `PATH` instead of `C:\ffmpeg\bin\ffmpeg.exe`.
5. Later cleanup, with review only: update KiCad plot/output metadata to project-local `hardware/pcb/.../plots`.
6. Do not delete old C caches until the project has run successfully for at least one full demo session.

## What This Plan Does Not Do

- It does not delete old C drive caches.
- It does not move datasets, raw videos, models, gallery files, or PCB files.
- It does not modify WiFi passwords, RTSP passwords, or MQTT credentials.
- It does not print RTSP/WiFi passwords in verification logs.
