import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).parent.parent  # python_cv/

if load_dotenv is not None:
    load_dotenv(ROOT / ".env")

MODELS_DIR = ROOT / "models"
GALLERY_DIR = ROOT / "data" / "gallery"
DB_PATH = ROOT / "outputs" / "warehouse.db"
LOG_DIR = ROOT / "outputs" / "logs"
DEBUG_DIR = ROOT / "outputs" / "debug_frames"

WAREHOUSE_DATASET_ROOT = Path(
    os.getenv("WAREHOUSE_DATASET_ROOT", "D:/warehouse_dataset")
).expanduser()
DATASET_CROPS_ROOT = Path(
    os.getenv("DATASET_CROPS_ROOT", str(WAREHOUSE_DATASET_ROOT / "dataset_crops"))
).expanduser()
RAW_VIDEO_ROOT = Path(
    os.getenv("RAW_VIDEO_ROOT", str(WAREHOUSE_DATASET_ROOT / "raw_videos_full"))
).expanduser()
REVIEW_CROP_ROOT = Path(
    os.getenv("REVIEW_CROP_ROOT", str(WAREHOUSE_DATASET_ROOT / "review_crops_reid_yoloseg"))
).expanduser()
IMOU_SD_VIDEO_ROOT = Path(
    os.getenv("IMOU_SD_VIDEO_ROOT", str(WAREHOUSE_DATASET_ROOT / "raw_videos_from_imou_sd"))
).expanduser()
REVIEW_CROPS_FROM_RAW_ROOT = Path(
    os.getenv("REVIEW_CROPS_FROM_RAW_ROOT", str(WAREHOUSE_DATASET_ROOT / "review_crops_from_raw"))
).expanduser()
RAW_VIDEO_2K_ROOT = Path(
    os.getenv("RAW_VIDEO_2K_ROOT", str(WAREHOUSE_DATASET_ROOT / "raw_videos"))
).expanduser()

YOLO_WEIGHTS = ROOT / "yolo11n-seg.pt"

if not YOLO_WEIGHTS.exists():
    YOLO_WEIGHTS = ROOT.parent / "yolo11n-seg.pt"

# ============================================================
# MQTT
# ============================================================

MQTT_BROKER = os.getenv("MQTT_BROKER", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "warehouse/rfid/scan")
MQTT_CLIENT = os.getenv("MQTT_CLIENT", "pc-warehouse-server")
ENABLE_MQTT = os.getenv("ENABLE_MQTT", "False").lower() == "true"

# ============================================================
# CAMERA
# ============================================================
# Zone 1: phone webcam / USB camera
# Zone 2: Imou RTSP hoặc webcam tạm khi demo không có IP camera
#
# Realtime nên dùng subtype=1 cho nhẹ.
# Khi thu data nét hơn có thể đổi CAM_ZONE2_RTSP subtype=0 trong .env.

ENABLE_ZONE1 = os.getenv("ENABLE_ZONE1", "True").lower() == "true"
ENABLE_ZONE2 = os.getenv("ENABLE_ZONE2", "True").lower() == "true"

CAM_ZONE1_INDEX = int(os.getenv("CAM_ZONE1_INDEX", "0"))
CAM_ZONE1_ROTATE = os.getenv("CAM_ZONE1_ROTATE", "90cw").lower()

CAM_ZONE2_TYPE = os.getenv("CAM_ZONE2_TYPE", "webcam").lower()
CAM_ZONE2_INDEX = int(os.getenv("CAM_ZONE2_INDEX", "1"))
CAM_ZONE2_RTSP = os.getenv(
    "CAM_ZONE2_RTSP",
    "rtsp://username:password@camera-ip:554/cam/realmonitor?channel=1&subtype=1",
)
CAMERA_STREAMS = {
    "zone1": {
        "name": "Zone 1 - Phone Webcam",
        "type": "webcam",
        "source": CAM_ZONE1_INDEX,
        "enabled": ENABLE_ZONE1,
    },
    "zone2": {
        "name": "Zone 2",
        "type": CAM_ZONE2_TYPE,
        "source": CAM_ZONE2_INDEX if CAM_ZONE2_TYPE == "webcam" else CAM_ZONE2_RTSP,
        "enabled": ENABLE_ZONE2,
    },
}


# ============================================================
# YOLO
# ============================================================

YOLO_CONF = float(os.getenv("YOLO_CONF", "0.4"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.5"))
YOLO_CLASSES = [0]  # person only
YOLO_IMG_SIZE = int(os.getenv("YOLO_IMG_SIZE", "640"))


# ============================================================
# TRACKING
# ============================================================

TRACKER_TYPE = os.getenv("TRACKER_TYPE", "bytetrack")  # bytetrack | botsort


# ============================================================
# ENTRY LINE ZONE 1
# ============================================================
# Pixel Y chia frame webcam làm 2 nửa. Chỉnh theo thực tế khi setup.

ENTRY_LINE_Y = int(os.getenv("ENTRY_LINE_Y", "250"))


# ============================================================
# FUSION RFID + RE-ID
# ============================================================

FUSION_TIME_WINDOW = float(os.getenv("FUSION_TIME_WINDOW", "10.0"))
FUSION_THRESHOLD = float(os.getenv("FUSION_THRESHOLD", "0.30"))
FUSION_WEIGHT_TIME = float(os.getenv("FUSION_WEIGHT_TIME", "0.4"))
FUSION_WEIGHT_REID = float(os.getenv("FUSION_WEIGHT_REID", "0.6"))


# ============================================================
# RE-ID
# ============================================================

REID_MODEL_NAME = os.getenv("REID_MODEL_NAME", "osnet_x1_0")
REID_INPUT_SIZE = (128, 256)  # (W, H)
REID_GALLERY_SIZE = int(os.getenv("REID_GALLERY_SIZE", "50"))
REID_MATCH_THRESHOLD = float(os.getenv("REID_MATCH_THRESHOLD", "0.93"))
REID_MATCH_MARGIN = float(os.getenv("REID_MATCH_MARGIN", "0.015"))
REID_TOPK_MEAN = int(os.getenv("REID_TOPK_MEAN", "5"))
# Model fine-tuned ResNet50 V3 clean-val
REID_FINETUNED_PATH = os.getenv(
    "REID_FINETUNED_PATH",
    "models/reid_resnet50_v3_cleanval.pth",
)


# ============================================================
# RFID-TRIGGER VISUAL VERIFY (ZONE 1)
# ============================================================

RFID_VISUAL_MATCH_THRESHOLD   = float(os.getenv("RFID_VISUAL_MATCH_THRESHOLD",   "0.65"))
ZONE1_BEST_PERSON_MAX_AGE_SEC = float(os.getenv("ZONE1_BEST_PERSON_MAX_AGE_SEC", "2.0"))
ZONE1_ENTRY_LINE_RATIO        = float(os.getenv("ZONE1_ENTRY_LINE_RATIO", "0.5"))
# Hướng đi tính là "VÀO KHO": "lr" = trái→phải (trái vạch = ngoài cửa, phải vạch = trong kho);
# "rl" = phải→trái. Chỉ hướng vào kho mới kích hoạt fusion / RFID / phát hiện intruder.
ZONE1_ENTRY_DIRECTION         = os.getenv("ZONE1_ENTRY_DIRECTION", "lr").lower()
ZONE1_FACE_REQUIRED           = os.getenv("ZONE1_FACE_REQUIRED", "false").lower() == "true"
ZONE1_FACE_UPSCALE            = os.getenv("ZONE1_FACE_UPSCALE", "true").lower() == "true"
ZONE1_FACE_UPSCALE_MIN_W      = int(os.getenv("ZONE1_FACE_UPSCALE_MIN_W", "200"))


# ============================================================
# INSIGHTFACE (OPTIONAL, ZONE 2 HYBRID)
# ============================================================

ENABLE_INSIGHTFACE = os.getenv("ENABLE_INSIGHTFACE", "False").lower() == "true"
FACE_MODEL_NAME = os.getenv("FACE_MODEL_NAME", "buffalo_l")
FACE_DET_SIZE = tuple(
    int(x.strip())
    for x in os.getenv("FACE_DET_SIZE", "640,640").replace("x", ",").split(",")
    if x.strip()
)
if len(FACE_DET_SIZE) != 2:
    FACE_DET_SIZE = (640, 640)
FACE_DET_THRESH = float(os.getenv("FACE_DET_THRESH", "0.35"))
FACE_GALLERY_PATH = ROOT / "data" / "face_gallery" / "insightface_gallery.pkl"
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.38"))
FACE_MATCH_MARGIN = float(os.getenv("FACE_MATCH_MARGIN", "0.09"))


# ============================================================
# UID -> EMPLOYEE MAPPING
# ============================================================
# Cập nhật khi enroll thẻ mới.
# Hiện tại model V3 chỉ học 5 class NV001–NV005.

UID_MAP = {
    "03F76328": "NV001",  # Mẫn (thẻ gốc NV001)
    # "23FCDA26": "NV002",  # Mai — ĐÃ GỠ để test kịch bản thẻ lạ (unknown_uid)
    "F3AC7128": "NV003",  # Minh
    "438DFE27": "NV005",  # Thẻ gốc của chị (Dung) — gán lại cho Mến (NV005) do mất thẻ NV005
    "95B0F605": "NV005",  # Mến (thẻ gốc NV005 — đã mất, giữ lại phòng khi tìm được)
}

EMPLOYEE_NAMES = {
    "NV001": "Mẫn",
    "NV002": "Mai",
    "NV003": "Minh",
    "NV004": "Dung",
    "NV005": "Mến",
}


# ============================================================
# DASHBOARD
# ============================================================

FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"


# ============================================================
# ENSURE OUTPUT DIRS EXIST
# ============================================================

for _d in (GALLERY_DIR, DB_PATH.parent, LOG_DIR, DEBUG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
