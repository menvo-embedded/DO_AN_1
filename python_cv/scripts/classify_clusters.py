import cv2, sys, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATASET_CROPS_ROOT

base = DATASET_CROPS_ROOT

ORDER = [
    ('NV001', 'Bo Man'),
    ('NV006', 'Me Mai'),
    ('NV002', 'Anh Minh'),
    ('NV004', 'Chi Dung'),
    ('NV007', 'Ban (toi)'),
]
NV_MAP = {'1':'NV001','2':'NV006','3':'NV002','4':'NV004','5':'NV007'}
NAME   = {'NV001':'Bo Man','NV006':'Me Mai','NV002':'Anh Minh','NV004':'Chi Dung','NV007':'Ban(toi)'}

for nv, label in ORDER:
    folder = base/nv
    if not folder.exists():
        print(f'{label}: folder not found, skip')
        continue
    imgs = sorted(folder.glob('*.jpg'))
    total = len(imgs)
    kept = deleted = moved = hard = 0
    print(f'\n=== {label} ({nv}) - {total} images ===')
    print('SPACE=dung | X=kho(2nguoi) | 1=Bo 2=Me 3=Anh 4=Chi 5=Ban | d=xoa | q=skip | ESC=thoat')

    for i, p in enumerate(imgs):
        img = cv2.imread(str(p))
        if img is None: continue

        display = cv2.resize(img, (280, 520))
        cv2.putText(display, f'{label}',
                    (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(display, f'{i+1}/{total} kept:{kept} del:{deleted} hard:{hard} mv:{moved}',
                    (5,52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,0), 1)
        cv2.putText(display, 'SPC=dung | X=kho(2nguoi)',
                    (5,445), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,255,255), 1)
        cv2.putText(display, '1=Bo 2=Me 3=Anh 4=Chi 5=Ban',
                    (5,462), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0,200,255), 1)
        cv2.putText(display, 'd=xoa | q=skip folder | ESC=thoat',
                    (5,480), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100,100,255), 1)
        cv2.imshow('Filter', display)

        while True:
            key = cv2.waitKey(0) & 0xFF
            ch  = chr(key) if key < 128 else ''

            if key == 27:  # ESC
                cv2.destroyAllWindows()
                print(f'Stopped. kept={kept} deleted={deleted} hard={hard} moved={moved}')
                sys.exit()

            elif key == ord('q'):  # skip folder
                cv2.destroyAllWindows()
                break

            elif key == 32:  # SPACE = giữ nguyên
                kept += 1
                break

            elif key == ord('d'):  # xóa
                p.unlink()
                deleted += 1
                break

            elif key == ord('x'):  # hard example (2+ người)
                dst = base / 'mixed' / nv
                dst.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(dst/p.name))
                hard += 1
                break

            elif ch in NV_MAP:  # chuyển folder
                dst_nv = NV_MAP[ch]
                if dst_nv != nv:
                    dst = base/dst_nv
                    dst.mkdir(exist_ok=True)
                    shutil.move(str(p), str(dst/p.name))
                    moved += 1
                else:
                    kept += 1
                break

    cv2.destroyAllWindows()
    remaining = len(list((base/nv).glob('*.jpg')))
    print(f'{label}: kept={kept} del={deleted} hard={hard} mv={moved} | remaining={remaining}')

# Tong ket
print('\n=== TONG KET ===')
for nv, label in ORDER:
    d = base/nv
    count = len(list(d.glob('*.jpg'))) if d.exists() else 0
    mixed_count = len(list((base/'mixed'/nv).glob('*.jpg'))) if (base/'mixed'/nv).exists() else 0
    print(f'  {label} ({nv}): {count} clean + {mixed_count} hard examples')

print('\nALL DONE!')
