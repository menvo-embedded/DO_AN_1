import argparse
import random
from pathlib import Path

import cv2
import numpy as np


PEOPLE = ["NV001_Mến", "NV002_Đức"]

L1 = "L1_Dễ_chính_diện_rõ_mặt"
L2 = "L2_Bình_thường_hơi_nghiêng"
L3 = "L3_Biến_thể_góc_mặt"
L4 = "L4_Khó_hợp_lệ"
L5 = "L5_Rất_khó_hợp_lệ"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img):
    ext = ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))
    return ok


def laplacian_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def collect_source_images(person_dir: Path):
    src_dirs = [person_dir / L1, person_dir / L2, person_dir / L3]
    imgs = []

    for d in src_dirs:
        if not d.exists():
            continue

        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                img = imread_unicode(p)
                if img is None:
                    continue
                sharp = laplacian_sharpness(img)
                imgs.append((p, sharp))

    imgs = sorted(imgs, key=lambda x: x[1], reverse=True)
    return imgs


def gaussian_blur(img, k=5):
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), 0)


def motion_blur(img, k=9, angle=0):
    k = max(3, int(k))
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    kernel /= k

    center = (k // 2, k // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (k, k))
    return cv2.filter2D(img, -1, kernel)


def adjust_brightness_contrast(img, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def add_noise(img, sigma=8):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def downscale_upscale(img, scale=0.65):
    h, w = img.shape[:2]
    nw = max(8, int(w * scale))
    nh = max(8, int(h * scale))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def jpeg_compress(img, quality=55):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, enc = cv2.imencode(".jpg", img, encode_param)
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else img


def random_shadow(img, strength=0.35):
    h, w = img.shape[:2]
    overlay = img.copy().astype(np.float32)

    x1 = random.randint(0, max(1, w // 2))
    x2 = random.randint(max(1, w // 2), w)

    mask = np.zeros((h, w), dtype=np.float32)
    cv2.rectangle(mask, (x1, 0), (x2, h), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)

    factor = 1.0 - mask[..., None] * strength
    out = overlay * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def augment_l4(img):
    out = img.copy()

    ops = [
        lambda x: gaussian_blur(x, random.choice([3, 5])),
        lambda x: motion_blur(x, random.choice([5, 7]), random.choice([0, 10, -10, 20, -20])),
        lambda x: adjust_brightness_contrast(x, alpha=random.uniform(0.85, 1.00), beta=random.randint(-20, 0)),
        lambda x: add_noise(x, sigma=random.uniform(3, 8)),
        lambda x: downscale_upscale(x, scale=random.uniform(0.70, 0.85)),
        lambda x: jpeg_compress(x, quality=random.randint(55, 75)),
        lambda x: random_shadow(x, strength=random.uniform(0.15, 0.30)),
    ]

    random.shuffle(ops)
    for op in ops[:random.randint(2, 4)]:
        out = op(out)

    return out


def augment_l5(img):
    out = img.copy()

    ops = [
        lambda x: gaussian_blur(x, random.choice([5, 7])),
        lambda x: motion_blur(x, random.choice([7, 9, 11]), random.choice([0, 15, -15, 25, -25, 35, -35])),
        lambda x: adjust_brightness_contrast(x, alpha=random.uniform(0.70, 0.95), beta=random.randint(-45, -10)),
        lambda x: add_noise(x, sigma=random.uniform(8, 16)),
        lambda x: downscale_upscale(x, scale=random.uniform(0.45, 0.70)),
        lambda x: jpeg_compress(x, quality=random.randint(35, 55)),
        lambda x: random_shadow(x, strength=random.uniform(0.25, 0.45)),
    ]

    random.shuffle(ops)
    for op in ops[:random.randint(3, 5)]:
        out = op(out)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"D:\warehouse_dataset_v2\face_data_tieng_viet")
    parser.add_argument("--l4-per-person", type=int, default=70)
    parser.add_argument("--l5-per-person", type=int, default=40)
    parser.add_argument("--top-pool", type=int, default=220)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    root = Path(args.root)

    for person in PEOPLE:
        person_dir = root / person
        l4_dir = person_dir / L4
        l5_dir = person_dir / L5

        l4_dir.mkdir(parents=True, exist_ok=True)
        l5_dir.mkdir(parents=True, exist_ok=True)

        src_imgs = collect_source_images(person_dir)

        if not src_imgs:
            print(f"[WARN] Không có ảnh nguồn cho {person}")
            continue

        top_pool = src_imgs[: min(len(src_imgs), args.top_pool)]

        print("=" * 80)
        print(f"Người: {person}")
        print(f"Số ảnh nguồn: {len(src_imgs)}")
        print(f"Top pool dùng để augment: {len(top_pool)}")

        created_l4 = 0
        created_l5 = 0

        for i in range(args.l4_per_person):
            src_path, sharp = random.choice(top_pool)
            img = imread_unicode(src_path)
            if img is None:
                continue

            aug = augment_l4(img)
            out_name = f"SYN_L4_{src_path.stem}_aug{i:04d}.jpg"
            out_path = l4_dir / out_name
            if imwrite_unicode(out_path, aug):
                created_l4 += 1

        for i in range(args.l5_per_person):
            src_path, sharp = random.choice(top_pool)
            img = imread_unicode(src_path)
            if img is None:
                continue

            aug = augment_l5(img)
            out_name = f"SYN_L5_{src_path.stem}_aug{i:04d}.jpg"
            out_path = l5_dir / out_name
            if imwrite_unicode(out_path, aug):
                created_l5 += 1

        print(f"Đã tạo L4: {created_l4}")
        print(f"Đã tạo L5: {created_l5}")

    print("=" * 80)
    print("DONE augmentation L4/L5")


if __name__ == "__main__":
    main()