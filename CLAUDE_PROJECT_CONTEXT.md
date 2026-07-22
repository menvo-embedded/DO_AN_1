# Context project cho Claude AI

## 1. Tên đề tài và mục tiêu đồ án

**Tên đề tài:** Hệ thống kiểm soát ra vào kho bằng RFID kết hợp Computer Vision.

**Mục tiêu:** Xây dựng prototype giám sát nhân viên trong kho bằng cách kết hợp:

- ESP32 + RFID RC522 để xác thực lượt quẹt thẻ tại cổng vào.
- Camera + YOLO để phát hiện người.
- ByteTrack để giữ `track_id` theo từng camera.
- Body Re-ID và optional InsightFace để nhận diện lại nhân viên khi di chuyển qua các zone.
- Fusion Layer để ghép sự kiện RFID với track camera.
- SQLite + Flask Dashboard để lưu log và hiển thị trạng thái realtime.

Entrypoint PC-side hiện tại là `python_cv/main.py`. Thư mục `warehouse-access-rfid-cv/` nằm lồng bên trong root là bản cũ/stale, không nên dùng làm nguồn chính nếu chưa review.

## 2. Bài toán thực tế hệ thống đang giải quyết

Trong kho có nhiều nhân viên đi qua cổng và di chuyển giữa các khu vực. Chỉ dùng RFID thì biết ai đã quẹt thẻ nhưng không chắc người đó có thật sự đi qua cổng hay có quẹt hộ. Chỉ dùng camera/Re-ID thì dễ nhầm người khi góc nhìn xấu, che mặt, ánh sáng kém hoặc nhiều người giống nhau.

Hệ thống giải quyết bài toán bằng cách:

- Nhân viên quẹt RFID tại Zone 1.
- Camera Zone 1 xác nhận có người đi qua entry line sau khi quẹt.
- Fusion Layer ghép sự kiện RFID gần nhất với track người đi qua.
- Khi người sang Zone 2/Zone 3, Re-ID nhận diện lại để cập nhật vị trí hiện tại.
- Dashboard hiển thị lượt vào, RFID events, anomaly và presence realtime.

Các anomaly chính: UID không hợp lệ, có người đi qua cổng nhưng không quẹt RFID, quẹt RFID nhưng không đi qua cổng, hoặc nghi vấn quẹt hộ khi fusion score thấp.

## 3. Kiến trúc tổng thể

```text
RFID Card
  -> ESP32 + RC522 + LCD + buzzer + servo gate
  -> MQTT topic warehouse/rfid/scan
  -> Python MQTT Listener
  -> Fusion Layer

Camera Zone 1
  -> YOLO person detection
  -> ByteTrack
  -> Entry line crossing
  -> Fusion Layer ghép RFID event với camera track
  -> SQLite entry_log / anomaly_log / presence_log

Camera Zone 2 / Zone 3
  -> YOLO person detection
  -> ByteTrack
  -> Body Re-ID / InsightFace hybrid
  -> Fusion Layer cập nhật identity và zone
  -> SQLite presence_log

Dashboard Flask
  -> REST API đọc SQLite
  -> Web UI realtime entries / RFID / anomalies / presence
```

Các thành phần chính:

| Thành phần | Vai trò | Code chính |
|---|---|---|
| RFID | Xác thực thẻ, mở cổng, publish MQTT | `firmware/src/main.cpp` |
| ESP32 | Điều khiển RC522, LCD, buzzer, servo, WiFi/MQTT | `firmware/` |
| MQTT | Nhận JSON RFID từ ESP32 | `python_cv/mqtt/mqtt_listener.py` |
| Camera | Lấy frame webcam/RTSP theo zone | `python_cv/detection/` |
| YOLO | Detect người, class person only | `ultralytics.YOLO`, weights `yolo11n-seg.pt` |
| ByteTrack | Tracking người theo frame | `python_cv/tracking/tracker.py` |
| Re-ID | Body embedding + cosine similarity top-k mean | `python_cv/reid/reid_engine.py` |
| InsightFace | Optional face signal cho Zone 2 hybrid | `python_cv/reid/face_insightface_engine.py` |
| Fusion Layer | Ghép RFID, crossing, Re-ID, anomaly | `python_cv/fusion/fusion_layer.py` |
| Database | SQLite schema/query/migration nhẹ | `python_cv/database/` |
| Dashboard | Flask UI và API realtime | `python_cv/dashboard/app.py` |

## 4. Cấu trúc project hiện tại

Root hiện tại: `D:\warehouse-access-rfid-cv`

Các thư mục quan trọng:

- `python_cv/`: pipeline PC-side chính.
- `firmware/`: firmware ESP32 PlatformIO.
- `hardware/`: PCB KiCad, footprint, 3D model, Gerber.
- `docs/`: hướng dẫn demo và tài liệu nộp.
- `scripts/`: PowerShell script migrate/verify dataset path.
- `archive/`: file cũ/backup.
- `warehouse-access-rfid-cv/`: bản nested/stale, không phải active pipeline.

## 5. Mô tả từng module code chính

| Module | File/thư mục | Mô tả trạng thái |
|---|---|---|
| Config | `python_cv/config/settings.py` | Central config, đọc thêm `python_cv/.env` nếu có. Định nghĩa model path, MQTT, camera, YOLO, Re-ID threshold, UID map, dashboard. Có default RTSP cần sanitize khi chia sẻ. |
| Detection | `python_cv/detection/camera_zone1.py` | Zone 1 dùng webcam index, YOLO + ByteTrack, xoay frame 90 độ, detect crossing qua entry line dọc giữa frame, gọi `fusion.on_entry_crossing(...)`. |
| Detection | `python_cv/detection/camera_zone2.py` | Zone 2 dùng RTSP hoặc webcam, resize 640x480, YOLO + ByteTrack, lọc crop chất lượng, gọi `fusion.identify_zone2(...)`. Có reconnect RTSP. |
| Detection | `python_cv/detection/camera_zone3.py` | Code Zone 3 có tồn tại, gọi `fusion.identify_zone3(...)`, nhưng hiện chưa được import/chạy trong `main.py` và `settings.py` chưa định nghĩa `CAM_ZONE3_RTSP`. |
| Tracking | `python_cv/tracking/tracker.py` | Wrapper `supervision.ByteTrack`, chỉ giữ detection class person. |
| Fusion | `python_cv/fusion/fusion_layer.py` | Quản lý pending RFID, crossings, Hungarian matching, track lock Zone 2, Re-ID zone, anomaly timeout. Có zone-aware key dạng `zone2:1`. |
| Re-ID | `python_cv/reid/reid_engine.py` | Body Re-ID. Load fine-tuned ResNet50 v3 nếu có, embedding 2048D; fallback OSNet nếu thiếu model. Matching dùng cosine similarity top-k mean. |
| Gallery | `python_cv/reid/gallery.py` | Load/save body gallery pickle tại `python_cv/data/gallery/gallery.pkl`; mỗi employee giữ deque giới hạn theo `REID_GALLERY_SIZE`. |
| Face Re-ID | `python_cv/reid/face_insightface_engine.py` | Optional InsightFace `buffalo_sc`, face gallery pickle, top-k mean, threshold/margin riêng. |
| Training | `python_cv/reid/trainer.py` | Script train/eval OSNet cũ theo dataset crop. Hiện runtime chính dùng ResNet50 fine-tuned trong `reid_engine.py`. |
| Database | `python_cv/database/database.py`, `models.py` | SQLite tại `python_cv/outputs/warehouse.db`; bảng `rfid_events`, `entry_log`, `anomaly_log`, `presence_log`; có migration nhẹ cho schema cũ. |
| Dashboard | `python_cv/dashboard/app.py` | Flask app, API `/api/entries`, `/api/anomalies`, `/api/rfid`, `/api/presence`, HTML dashboard refresh định kỳ. |
| MQTT | `python_cv/mqtt/mqtt_listener.py` | Paho MQTT listener, parse JSON payload RFID, thêm `dt=datetime.fromisoformat(...)`, callback vào Fusion Layer. |
| Tools | `python_cv/tools/` | Script collect/debug/calibrate data: build gallery, crop từ video, audit dataset, calibrate threshold/margin, synthetic tests, debug webcam hybrid. |
| Scripts | `python_cv/scripts/` | Script cũ/tiện ích như diagnose score, auto crop, evaluate pipeline. |
| Tests | `python_cv/tests/` | Test camera, RTSP, YOLO, InsightFace, webcam Re-ID. Một số test cũ/placeholder; có file test RTSP chứa credential trong source nên không đưa nguyên trạng cho bên ngoài. |
| Main | `python_cv/main.py` | Khởi tạo DB, ReIDEngine, Gallery, optional InsightFace, FusionLayer, optional MQTT, Zone 2 thread, dashboard thread, Zone 1 main thread. |

## 6. Luồng hoạt động chính

### Luồng RFID + Zone 1

1. Nhân viên quẹt RFID tại cổng Zone 1.
2. ESP32 đọc UID, tìm employee ID trong firmware, mở servo gate và publish MQTT JSON.
3. `RFIDListener` nhận event, gọi `FusionLayer.on_rfid_event(event)`.
4. Fusion ghi `rfid_events`, map UID sang employee theo `UID_MAP`, đưa employee vào `_pending`.
5. Camera Zone 1 phát hiện người bằng YOLO, ByteTrack gán `track_id`.
6. Khi tâm bbox đi qua entry line, Zone 1 cắt crop người và gọi `on_entry_crossing("zone1", track_id, crop, timestamp)`.
7. Fusion trích body embedding, ghép pending RFID với crossing bằng Hungarian matching.
8. Nếu score đủ ngưỡng, ghi `entry_log`, update `presence_log`, update gallery bằng embedding mới.
9. Nếu không match hoặc timeout, ghi anomaly như `no_rfid_intruder` hoặc `rfid_no_crossing`.

### Luồng Zone 2 / Zone 3

1. Camera Zone 2 phát hiện/tracking người.
2. Với crop đủ chất lượng, gọi `FusionLayer.identify_zone2(...)`.
3. Nếu bật InsightFace, Zone 2 dùng hybrid:
   - Face MATCH thì ưu tiên face và lock track trong vài giây.
   - NO_FACE thì hiện tại body fallback bị ignore để giảm nhận nhầm.
   - UNKNOWN thì chỉ dùng body nếu body rất strict.
4. Nếu không bật InsightFace, Zone 2 dùng body Re-ID thuần.
5. Khi xác nhận identity, Fusion update `presence_log` với zone hiện tại.
6. Zone 3 hiện có code riêng `identify_zone3(...)`, không còn gọi tạm `identify_zone2`, nhưng chưa được nối vào `main.py` và thiếu config `CAM_ZONE3_RTSP`.

## 7. Trạng thái hiện tại của project

### Phần đã chạy được / tương đối hoàn thiện

- Pipeline chính `python_cv/main.py` đã có cấu trúc chạy được cho demo CV-only hoặc RFID+CV tùy `.env`.
- Zone 1: webcam + YOLO + ByteTrack + entry line crossing.
- Zone 2: webcam/RTSP + YOLO + ByteTrack + body Re-ID hoặc InsightFace hybrid.
- Fusion Layer: pending RFID, Hungarian matching, anomaly timeout, zone-aware track key.
- SQLite: schema và query dashboard đã có.
- Dashboard Flask: có UI realtime entries/RFID/anomalies/presence tại `http://localhost:5000`.
- Dataset path đã chuyển sang default `D:/warehouse_dataset` và có script migrate/verify.
- Body gallery hiện có tại `python_cv/data/gallery/gallery.pkl`.

### Phần còn lỗi/chưa hoàn thiện

- Zone 3 chưa được nối vào `main.py`; `settings.py` chưa có `ENABLE_ZONE3` và `CAM_ZONE3_RTSP`.
- `python_cv/detection/camera_zone3.py` nếu import trực tiếp sẽ phụ thuộc `CAM_ZONE3_RTSP` chưa định nghĩa.
- Firmware và Python UID mapping chưa đồng bộ hoàn toàn: firmware có NV001-NV007, Python hiện chỉ map NV001-NV005.
- Firmware hiện có hardcoded WiFi/MQTT secret trong source; phải sanitize trước khi nộp/chia sẻ.
- `settings.py` có default RTSP URL chứa credential; khi viết tài liệu phải mask dạng `rtsp://admin:***@...`.
- `python_cv/app.py` là dashboard copy cũ, không phải dashboard chính. Dashboard chính là `python_cv/dashboard/app.py`.
- Một số file README cũ bị mojibake tiếng Việt.
- Nhiều backup/debug/output/model/gallery/raw data chưa sạch để nộp nguyên trạng.

### Camera zone hiện có

- Zone 1: `CAM_ZONE1_INDEX`, default trong code hiện là webcam index `3`; `.env.example` demo dùng `0`.
- Zone 2: hỗ trợ `CAM_ZONE2_TYPE=rtsp|webcam`; default code là RTSP, `.env.example` demo là webcam. RTSP phải được mask, không ghi password.
- Zone 3: có file `camera_zone3.py` nhưng chưa có config runtime trong `settings.py` và chưa được main chạy.

### Model Re-ID đang dùng

- Body Re-ID runtime ưu tiên: `python_cv/models/reid_resnet50_v3_cleanval.pth`.
- Kiến trúc runtime: ResNet50 bỏ FC, output embedding 2048D, normalize L2.
- Fallback nếu thiếu fine-tuned model: `torchreid` model `osnet_x1_0`.
- YOLO weights: `python_cv/yolo11n-seg.pt` nếu có, nếu không fallback root `yolo11n-seg.pt`.
- InsightFace model: `buffalo_sc`, model files có trong `python_cv/models/insightface/buffalo_sc/`.

### Gallery đang dùng

Body gallery hiện tại:

| Employee | Số embedding | Dimension |
|---|---:|---:|
| NV001 | 30 | 2048 |
| NV002 | 30 | 2048 |
| NV003 | 50 | 2048 |
| NV004 | 30 | 2048 |
| NV005 | 50 | 2048 |

Face gallery runtime path hiện tại là `python_cv/data/face_gallery/insightface_gallery.pkl`, nhưng file này chưa tồn tại tại thời điểm kiểm tra. Nếu bật `ENABLE_INSIGHTFACE=true`, engine sẽ load model nhưng gallery face rỗng/không có, nên cần build lại face gallery hoặc tắt InsightFace tùy demo.

## 8. Các lỗi/vấn đề kỹ thuật cần xử lý tiếp

### Re-ID nhận nhầm

Vấn đề người dùng đang quan tâm: Re-ID đôi khi nhận nhầm NV001 thành NV002. File calibration hiện tại không bắt đúng case này trực tiếp, nhưng có dấu hiệu confusion gần:

- Trong `reid_calibration_results.csv`, có case `NV001 -> NV003`, trong đó NV002 là second-best.
- Tổng calibration 250 ảnh có top1 accuracy khoảng `0.932`.
- Với config hiện tại `REID_MATCH_THRESHOLD=0.93`, `REID_MATCH_MARGIN=0.015`, summary có 210 correct confirm, 5 wrong confirm, 35 unknown.

Cần Claude ưu tiên kiểm tra:

- Logic `match_score(...)` trong `reid_engine.py`: top-k mean với `REID_TOPK_MEAN=5`.
- Logic `identify(...)`: `best_score >= threshold` và `best_score - second_score >= margin`.
- Các threshold hiện tại:
  - `REID_MATCH_THRESHOLD=0.93`
  - `REID_MATCH_MARGIN=0.015`
  - `REID_TOPK_MEAN=5`
  - Zone 2 strict body threshold trong Fusion: `0.95`, margin `0.035`
- Chất lượng gallery NV001/NV002: crop có bị lẫn người, lẫn góc camera, ảnh quá giống hoặc background bias không.
- Rebuild gallery sạch hơn hoặc tách gallery theo zone/camera nếu cần.

### Zone 3

- Hiện `camera_zone3.py` gọi `fusion.identify_zone3(track_id, crop)`, không còn gọi tạm `identify_zone2`.
- Nhưng Zone 3 chưa được wire vào `main.py`, chưa có `ENABLE_ZONE3`, chưa có `CAM_ZONE3_RTSP` trong `settings.py`.
- Nếu Claude tiếp tục Zone 3, cần thêm config và thread chạy riêng, không tự đổi kiến trúc nếu chưa được yêu cầu.

### Security/cleanup

- Không ghi nội dung `python_cv/.env`.
- Không ghi RTSP password.
- Không ghi WiFi password từ firmware.
- Nếu thấy credential trong source, thay bằng `***` khi viết tài liệu hoặc log gửi ra ngoài.

## 9. Các lệnh PowerShell quan trọng

Chạy từ repo root `D:\warehouse-access-rfid-cv` nếu không ghi khác.

### Tạo môi trường Python

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Tạo config local

```powershell
Copy-Item .env.example python_cv\.env
```

Sau đó sửa `python_cv\.env` trên máy local. Không commit/gửi file này.

### Chạy pipeline chính

```powershell
cd python_cv
python main.py
```

Dashboard:

```text
http://localhost:5000
```

### Demo CV-only gợi ý trong `.env`

```text
ENABLE_MQTT=false
ENABLE_ZONE1=false
ENABLE_ZONE2=true
CAM_ZONE2_TYPE=webcam
CAM_ZONE2_INDEX=0
ENABLE_INSIGHTFACE=false
```

Nếu muốn dùng InsightFace, cần đảm bảo face gallery tồn tại hoặc chấp nhận face engine không match.

### Demo RFID + CV end-to-end gợi ý

```text
ENABLE_MQTT=true
ENABLE_ZONE1=true
ENABLE_ZONE2=true
MQTT_BROKER=<broker-ip>
MQTT_TOPIC=warehouse/rfid/scan
```

### Flash/monitor firmware ESP32

```powershell
platformio run -d firmware -t upload
platformio device monitor -d firmware
```

Trước khi chia sẻ firmware, thay credential bằng placeholder hoặc chuyển sang config local không commit.

### Test camera theo settings

```powershell
cd python_cv
python tests\test_camera_streams_from_settings.py
```

### Kiểm tra môi trường InsightFace

```powershell
cd python_cv
python tests\check_insightface_env.py
```

### Diagnose Re-ID cho một ảnh crop

```powershell
cd python_cv
python scripts\diagnose_reid_scores.py <duong_dan_anh_crop>
```

### Calibrate threshold/margin Re-ID

```powershell
cd python_cv
python tools\calibrate_reid_threshold_margin.py --max-per-class 50
```

Output:

- `python_cv/outputs/reid_calibration/reid_calibration_results.csv`
- `python_cv/outputs/reid_calibration/reid_calibration_summary.csv`

### Verify dataset path

```powershell
.\scripts\verify_dataset_paths.ps1
```

### Migrate dataset từ F sang D nếu cần

```powershell
.\scripts\migrate_warehouse_dataset_F_to_D.ps1
```

Script này copy, không xóa dữ liệu nguồn.

## 10. File Claude nên ưu tiên đọc tiếp

Ưu tiên cao:

1. `README.md`
2. `docs/RUN_DEMO.md`
3. `python_cv/main.py`
4. `python_cv/config/settings.py`
5. `python_cv/fusion/fusion_layer.py`
6. `python_cv/reid/reid_engine.py`
7. `python_cv/reid/gallery.py`
8. `python_cv/detection/camera_zone1.py`
9. `python_cv/detection/camera_zone2.py`
10. `python_cv/detection/camera_zone3.py`
11. `python_cv/database/database.py`
12. `python_cv/dashboard/app.py`
13. `python_cv/mqtt/mqtt_listener.py`
14. `python_cv/tools/calibrate_reid_threshold_margin.py`
15. `python_cv/scripts/diagnose_reid_scores.py`

Đọc khi làm firmware/hardware:

1. `firmware/src/main.cpp`
2. `firmware/platformio.ini`
3. `firmware/config/device_config.example.h`
4. `hardware/pcb/warehouse_rfid_controller/`

Đọc khi cleanup/nộp đồ án:

1. `CLEANUP_AUDIT_REPORT.md`
2. `SUBMISSION_STRUCTURE_PROPOSAL.md`
3. `DATA_PATH_MIGRATION_REPORT.md`
4. `.env.example`

Không ưu tiên:

- `warehouse-access-rfid-cv/` nested stale copy.
- `*_backup_before_*.py`.
- `python_cv/app.py` dashboard copy cũ.
- `python_cv/outputs/` trừ khi đang debug log/calibration cụ thể.

## 11. Checklist việc cần làm tiếp

- [ ] Chốt demo mode: CV-only Zone 2 webcam hay RFID + Zone 1 + Zone 2 end-to-end.
- [ ] Sanitize firmware: bỏ WiFi/MQTT secret khỏi `firmware/src/main.cpp`.
- [ ] Sanitize RTSP default trong `python_cv/config/settings.py` hoặc đảm bảo dùng `.env`.
- [ ] Đồng bộ UID mapping giữa firmware và Python cho NV001-NV005 hoặc NV001-NV007.
- [ ] Debug case Re-ID nhận nhầm NV001/NV002 bằng `diagnose_reid_scores.py`.
- [ ] Kiểm tra lại `REID_TOPK_MEAN`, `REID_MATCH_THRESHOLD`, `REID_MATCH_MARGIN`.
- [ ] Audit/rebuild gallery sạch hơn nếu có crop lẫn người hoặc background bias.
- [ ] Quyết định dùng/tắt InsightFace; nếu dùng thì build `python_cv/data/face_gallery/insightface_gallery.pkl`.
- [ ] Nếu cần Zone 3, thêm `ENABLE_ZONE3`, `CAM_ZONE3_RTSP`, thread trong `main.py`, và test import.
- [ ] Dọn hoặc archive bản nested `warehouse-access-rfid-cv/` trước khi nộp.
- [ ] Loại khỏi bản nộp: `.env`, raw video, ảnh người thật, gallery pickle, SQLite log thật, model nặng nếu không có yêu cầu.
- [ ] Viết lại các README tiếng Việt bị mojibake nếu dùng trong báo cáo.
- [ ] Chạy thử `python main.py` và dashboard trước buổi demo.
- [ ] Lưu lại một số screenshot/log đã ẩn danh để làm bằng chứng demo.
