import cv2
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from config.settings import DATASET_CROPS_ROOT

base  = DATASET_CROPS_ROOT
NAMES = {'NV001':'Bo','NV002':'Me','NV003':'Anh','NV004':'Chi','NV005':'Toi'}

print('='*55)
print('KIEM TRA DATASET TRUOC KHI TRAIN')
print('='*55)

total_issues = 0

for nv in ['NV001','NV002','NV003','NV004','NV005']:
    clean_dir = base / nv
    hard_dir  = base / 'mixed' / nv

    issues = []
    stats  = defaultdict(int)

    for folder, tag in [(clean_dir, 'clean'), (hard_dir, 'hard')]:
        if not folder.exists():
            continue
        imgs = list(folder.glob('*.jpg'))
        stats[tag] = len(imgs)

        for p in imgs:
            img = cv2.imread(str(p))
            if img is None:
                issues.append(f'UNREADABLE: {p.name}')
                continue
            h, w = img.shape[:2]
            if h < 32 or w < 16:
                issues.append(f'TOO_SMALL ({w}x{h}): {p.name}')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lap  = cv2.Laplacian(gray, cv2.CV_64F).var()
            if lap < 20:
                issues.append(f'TOO_BLURRY (lap={lap:.0f}): {p.name}')
            if h > 0 and w > 0:
                ar = h / w
                if ar < 1.0 or ar > 5.0:
                    issues.append(f'BAD_RATIO ({ar:.1f}): {p.name}')

    total = stats['clean'] + stats['hard']
    print(f"\n{nv} ({NAMES[nv]}): {stats['clean']} clean + {stats['hard']} hard = {total}")

    if issues:
        print(f'  [WARN] {len(issues)} issues:')
        for iss in issues[:5]:
            print(f'    - {iss}')
        if len(issues) > 5:
            print(f'    ... va {len(issues)-5} loi khac')
        total_issues += len(issues)
    else:
        print('  [OK] Khong co loi')

print('\n' + '='*55)
print(f'Tong loi: {total_issues}')

# Class imbalance
print('\nClass distribution:')
counts = []
for nv in ['NV001','NV002','NV003','NV004','NV005']:
    clean = len(list((base/nv).glob('*.jpg'))) if (base/nv).exists() else 0
    hard  = len(list((base/'mixed'/nv).glob('*.jpg'))) if (base/'mixed'/nv).exists() else 0
    total = clean + hard
    counts.append(total)
    bar = '#' * (total // 50)
    print(f'  {nv} ({NAMES[nv]}): {bar} {total}')

mx, mn = max(counts), min(counts)
print(f'\nMax/Min ratio: {mx/mn:.1f}x')
if mx/mn > 5:
    print('[WARN] Mat can bang nghiem trong - WeightedSampler se xu ly')
else:
    print('[OK] Can bang chap nhan duoc')
