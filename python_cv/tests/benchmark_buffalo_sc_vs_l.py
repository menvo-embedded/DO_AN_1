# tests/benchmark_buffalo_sc_vs_l.py

import time
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ============================================================
# CONFIG
# ============================================================

IMG_PATH = "test_face.jpg"

# Test nhiều kích thước detect để xem mặt nhỏ/góc nghiêng có cải thiện không
DET_SIZES = [
    (320, 320),
    (480, 480),
    (640, 640),
]

N_RUNS = 20

MODELS = [
    "buffalo_sc",
    "buffalo_l",
]


def load_image(path: str):
    img = cv2.imread(path)

    if img is None:
        print(f"[ERROR] Cannot read image: {path}")
        print("Hãy đặt test_face.jpg trong thư mục python_cv hoặc sửa IMG_PATH.")
        raise SystemExit(1)

    print(f"Image: {path}, shape={img.shape}")
    return img


def benchmark(model_name: str, det_size: tuple[int, int], img):
    print("=" * 90)
    print(f"Testing model={model_name}, det_size={det_size}")

    # allowed_modules để test công bằng:
    # chỉ load detection + recognition, không load genderage/landmark thừa
    app = FaceAnalysis(
        name=model_name,
        allowed_modules=["detection", "recognition"],
    )

    # ctx_id=0: ưu tiên GPU nếu onnxruntime có CUDAExecutionProvider
    # nếu chưa có CUDA provider thì sẽ fallback CPU
    app.prepare(
        ctx_id=0,
        det_size=det_size,
    )

    # Warmup
    for _ in range(3):
        _ = app.get(img)

    times = []
    last_faces = []

    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        faces = app.get(img)
        dt = (time.perf_counter() - t0) * 1000.0

        times.append(dt)
        last_faces = faces

    mean_ms = np.mean(times)
    p95_ms = np.percentile(times, 95)
    min_ms = np.min(times)
    max_ms = np.max(times)

    print(f"faces={len(last_faces)}")
    print(f"mean={mean_ms:.1f}ms | p95={p95_ms:.1f}ms | min={min_ms:.1f}ms | max={max_ms:.1f}ms")

    if last_faces:
        for i, face in enumerate(last_faces):
            bbox = face.bbox.astype(int).tolist()
            det_score = float(face.det_score)

            print(
                f"face#{i}: "
                f"det_score={det_score:.3f}, "
                f"bbox={bbox}"
            )
    else:
        print("No face detected.")

    return {
        "model": model_name,
        "det_size": det_size,
        "faces": len(last_faces),
        "mean_ms": mean_ms,
        "p95_ms": p95_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "det_score": float(last_faces[0].det_score) if last_faces else 0.0,
    }


def main():
    img = load_image(IMG_PATH)

    print(f"N_RUNS={N_RUNS}")
    print(f"MODELS={MODELS}")
    print(f"DET_SIZES={DET_SIZES}")

    results = []

    for model_name in MODELS:
        for det_size in DET_SIZES:
            try:
                result = benchmark(model_name, det_size, img)
                results.append(result)
            except Exception as e:
                print("=" * 90)
                print(f"[ERROR] model={model_name}, det_size={det_size}")
                print(e)

    print("\n")
    print("#" * 90)
    print("SUMMARY")
    print("#" * 90)

    print(f"{'Model':<12} {'DetSize':<12} {'Faces':<6} {'Score':<8} {'Mean(ms)':<10} {'P95(ms)':<10}")
    print("-" * 90)

    for r in results:
        det_size_text = f"{r['det_size'][0]}x{r['det_size'][1]}"
        print(
            f"{r['model']:<12} "
            f"{det_size_text:<12} "
            f"{r['faces']:<6} "
            f"{r['det_score']:<8.3f} "
            f"{r['mean_ms']:<10.1f} "
            f"{r['p95_ms']:<10.1f}"
        )


if __name__ == "__main__":
    main()