import sys
from pathlib import Path
import cv2
from insightface.app import FaceAnalysis

IMG_PATH = Path("test_face.jpg")
if len(sys.argv) >= 2:
    IMG_PATH = Path(sys.argv[1])

if not IMG_PATH.exists():
    raise FileNotFoundError(IMG_PATH)

img = cv2.imread(str(IMG_PATH))
if img is None:
    raise RuntimeError(f"Cannot read image: {IMG_PATH}")

print("Image:", IMG_PATH)
print("Image shape:", img.shape)

app = FaceAnalysis(name="buffalo_sc", root="models/insightface")
app.prepare(ctx_id=0, det_size=(1024, 1024))

faces = app.get(img)

print(f"Faces detected: {len(faces)}")

for i, f in enumerate(faces, start=1):
    x1, y1, x2, y2 = f.bbox.astype(int)
    print(f"Face {i}: bbox={f.bbox}, score={f.det_score:.3f}, emb_dim={len(f.embedding)}")

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        img,
        f"face {i} {f.det_score:.2f}",
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

out = IMG_PATH.with_name(IMG_PATH.stem + "_face_test.jpg")
cv2.imwrite(str(out), img)
print("Saved:", out)
