import sys
import cv2
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from reid.reid_engine import ReIDEngine
from reid.gallery import Gallery

if len(sys.argv) < 2:
    print("Usage: python tools/test_crop_direct_reid.py <image_path>")
    raise SystemExit

img_path = Path(sys.argv[1])
img = cv2.imread(str(img_path))

if img is None:
    print("Cannot read:", img_path)
    raise SystemExit

reid = ReIDEngine()
gallery = Gallery()
gallery_data = gallery.all()

print("=" * 80)
print("DIRECT CROP RE-ID")
print("=" * 80)
print("Image:", img_path)
print("Shape:", img.shape)

query_emb = reid.get_embedding(img)

if query_emb is None:
    print("Cannot extract embedding.")
    raise SystemExit

rows = []

for emp_id, embeds in gallery_data.items():
    score = reid.match_score(query_emb, embeds)
    rows.append((emp_id, score))

rows.sort(key=lambda x: x[1], reverse=True)

best_id = rows[0][0] if rows else None
best_score = rows[0][1] if rows else 0.0
second_score = rows[1][1] if len(rows) > 1 else 0.0
margin = best_score - second_score
identified = reid.identify(img, gallery_data)

print("identified:", identified)
print("best_id:", best_id)
print("best_score:", f"{best_score:.4f}")
print("second_score:", f"{second_score:.4f}")
print("margin:", f"{margin:.4f}")
print()
print("ranking:")

for emp_id, score in rows:
    print(f"  {emp_id}: {score:.4f}")
