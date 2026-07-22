# Submission Structure Proposal

De xuat tao mot thu muc nop rieng, vi du `submission/warehouse-access-rfid-cv/`,
khong nop truc tiep repo hien tai nguyen trang.

```text
warehouse-access-rfid-cv/
  README.md
  requirements.txt
  .env.example
  platformio.ini

  firmware/
    README.md
    platformio.ini
    src/
      main.cpp
    config/
      device_config.example.h
    include/
      README.md
    lib/
      README.md

  python_cv/
    README.md
    main.py
    .env.example
    config/
      settings.py
      thresholds.yaml
    dashboard/
      app.py
      __init__.py
    database/
      database.py
      models.py
      __init__.py
    detection/
      camera_zone1.py
      camera_zone2.py
    fusion/
      fusion_layer.py
      __init__.py
    mqtt/
      mqtt_listener.py
      __init__.py
    reid/
      reid_engine.py
      gallery.py
      face_insightface_engine.py
      trainer.py
      __init__.py
    tracking/
      tracker.py
      __init__.py
    utils/
      logger.py
      __init__.py
    tools/
      build_face_gallery_from_dataset_crops.py
      calibrate_reid_threshold_margin.py
      capture_zone2_reid_debug.py
      offline_crop_reid_yoloseg.py
      standalone_webcam_hybrid_debug.py
    assets/
      sample_images/
        README.md

  hardware/
    pcb/
      warehouse_rfid_controller/
        warehouse_rfid_controller.kicad_pro
        warehouse_rfid_controller.kicad_sch
        warehouse_rfid_controller.kicad_pcb
        fp-lib-table
        custom_footprints.pretty/
        3d_models/
      gerber/

  docs/
    RUN_DEMO.md
    diagrams/
    proposal/
    report_assets/

  demo_assets/
    README.md
    screenshots/
    sample_logs/
    anonymized_samples/
    models/
      README.md
```

## Khong Dua Vao Ban Nop Source

- `python_cv/.env`
- `.pio/`, `firmware/.pio/`, `.pytest_cache/`, `.vscode/`, `__pycache__/`
- `warehouse-access-rfid-cv/` nested stale copy
- `archive/` tru khi giao vien yeu cau lich su debug
- `python_cv/outputs/` raw debug outputs
- `python_cv/data/dataset_raw/`
- `python_cv/data/dataset_crops/` neu chua an danh
- `python_cv/data/gallery/*.pkl`
- `python_cv/data/face_gallery/*.pkl`
- `python_cv/data/face_gallery/raw/`
- Backup/copy files: `*_backup_before_*.py`, `main_backup_before_*.py`,
  `crop_reid_from_video.py.py`, `code_review_*.txt`, `project_tree.txt`
- Model nang: `.pt`, `.pth`, `.onnx`, `.zip` neu khong co yeu cau nop offline

## Cach Xu Ly Model/Dataset Cho Demo

- Neu demo tren may ca nhan: giu model/dataset ngoai source zip va ghi duong dan
  setup trong `docs/RUN_DEMO.md`.
- Neu giao vien can chay offline: tao `demo_assets/models/README.md` ghi link
  tai model va checksum; chi nop model nho can thiet.
- Neu can sample anh/video: chi nop sample da an danh/blur mat, dung dung luong
  nho, khong nop raw data nguoi that.

## Diem Can Chot Truoc Khi Dong Goi

- Chot 1 pipeline demo: CV-only Zone 2 webcam hay RFID + Zone 1 + Zone 2.
- Chot UID mapping NV001-NV005 duy nhat giua firmware va Python.
- Chot Zone 3: bo khoi submission hoac cap nhat config/code day du.
- Sanitize firmware secrets truoc khi copy vao submission.
- Chay thu demo theo `docs/RUN_DEMO.md` tren may nop.
