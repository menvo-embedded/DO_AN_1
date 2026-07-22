# Data Path Migration Report

Audit date: 2026-05-07  
Goal: remove hard dependency on external `F:\warehouse_dataset` and use local
internal `D:\warehouse_dataset` through environment-driven settings.

## Summary

- Main pipeline no longer hardcodes `F:/warehouse_dataset/dataset_crops`.
- Dataset/video/review roots are now centralized in `python_cv/config/settings.py`.
- Default local data root is `D:/warehouse_dataset`.
- No raw video, image dataset, gallery pickle, or model file was moved, deleted,
  or edited.
- Migration and verification are provided as PowerShell scripts only.

## Files Modified

Core pipeline:

- `python_cv/config/settings.py`
- `python_cv/main.py`

Tools/scripts:

- `python_cv/check_dataset.py`
- `python_cv/reid/trainer.py`
- `python_cv/scripts/auto_crop_dataset.py`
- `python_cv/scripts/record_dataset.py`
- `python_cv/scripts/live_crop.py`
- `python_cv/scripts/classify_clusters.py`
- `python_cv/scripts/filter_reclassify.py`
- `python_cv/tools/build_face_gallery_from_dataset_crops.py`
- `python_cv/tools/calibrate_reid_threshold_margin.py`
- `python_cv/tools/crop_reid_from_video.py`
- `python_cv/tools/crop_reid_from_video.py.py`
- `python_cv/tools/crop_nv005_from_video.py`
- `python_cv/tools/offline_crop_persons_from_raw_videos.py`
- `python_cv/tools/offline_crop_reid_yoloseg.py`
- `python_cv/tools/preview_monitor_zone23.py`
- `python_cv/tools/record_raw_imou_2k.py`
- `python_cv/tools/record_raw_scene_imou.py`
- `python_cv/tools/record_raw_scene_imou_stable.py`
- `python_cv/tools/record_raw_scene_zone1.py`

Docs/config:

- `.env.example`
- `python_cv/.env.example`
- `README.md`
- `docs/RUN_DEMO.md`
- `CLEANUP_AUDIT_REPORT.md`

New scripts:

- `scripts/migrate_warehouse_dataset_F_to_D.ps1`
- `scripts/verify_dataset_paths.ps1`

## Environment Variables Added

Added to `.env.example` and `python_cv/.env.example`:

```text
WAREHOUSE_DATASET_ROOT=D:/warehouse_dataset
DATASET_CROPS_ROOT=D:/warehouse_dataset/dataset_crops
RAW_VIDEO_ROOT=D:/warehouse_dataset/raw_videos_full
REVIEW_CROP_ROOT=D:/warehouse_dataset/review_crops_reid_yoloseg
IMOU_SD_VIDEO_ROOT=D:/warehouse_dataset/raw_videos_from_imou_sd
```

Additional internal settings are available in `settings.py` for existing tools:

```text
REVIEW_CROPS_FROM_RAW_ROOT
RAW_VIDEO_2K_ROOT
```

These default under `WAREHOUSE_DATASET_ROOT`.

## Paths Replaced

Replaced hardcoded source/tool references to:

- `F:/warehouse_dataset/dataset_crops`
- `F:\warehouse_dataset\dataset_crops`
- `F:\warehouse_dataset\raw_videos_full`
- `F:\warehouse_dataset\review_crops_reid_yoloseg`
- `F:\warehouse_dataset\review_crops_from_raw`
- `F:\warehouse_dataset\raw_videos`

The code now resolves those through `config.settings` values, defaulting to
`D:/warehouse_dataset`.

## Remaining F Paths

Intentional remaining path:

- `scripts/migrate_warehouse_dataset_F_to_D.ps1` uses
  `F:\warehouse_dataset` as the copy source.

Not modified because they are old/archive/debug artifacts:

- `python_cv/main_backup_before_disable_mqtt.py`
- `python_cv/code_review_core_01.txt`
- `python_cv/project_tree.txt`

These should be archived or excluded from submission as already noted in
`CLEANUP_AUDIT_REPORT.md`.

## How to Run Migration

From repo root:

```powershell
.\scripts\migrate_warehouse_dataset_F_to_D.ps1
```

The script:

- Creates `D:\warehouse_dataset` if missing.
- Copies these folders when they exist:
  - `dataset_crops`
  - `raw_videos_full`
  - `review_crops_reid_yoloseg`
  - `raw_videos_from_imou_sd`
- Uses `robocopy` with logs under `D:\warehouse_dataset\_migration_logs`.
- Does not delete anything from `F:`.

## How to Verify

From repo root:

```powershell
.\scripts\verify_dataset_paths.ps1
```

The verify script checks:

- `D:\warehouse_dataset`
- `dataset_crops\NV001` through `NV005`
- videos under `raw_videos_full`
- image outputs under `review_crops_reid_yoloseg`
- `python_cv\data\gallery\gallery.pkl`

## Manual Checks / Risks

- `python_cv/.env` may still contain old local paths; update it manually or
  recreate it from `.env.example`.
- Some old tools still import `CAM_ZONE3_RTSP`, while current `settings.py` does
  not define Zone 3. This migration did not address Zone 3 config.
- `python_cv/tools/crop_reid_from_video.py.py` is a duplicate file and should be
  archived later; it was updated only to prevent stale F-path defaults.
- The migration scripts were created but not executed.
- No raw video, dataset image, gallery pickle, or model file was copied during
  this refactor.
