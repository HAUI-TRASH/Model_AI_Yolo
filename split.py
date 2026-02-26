from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "dataset"
DST = ROOT / "dataset_split"

KEEP_RATIO = 0.1  # giữ 10%
IMG_EXTS = (".jpg", ".jpeg", ".png")

print("SRC =", SRC)
print("DST =", DST)

def copy_pair(img_path: Path, lbl_dir: Path, out_img_dir: Path, out_lbl_dir: Path):
    # copy image
    shutil.copy(img_path, out_img_dir / img_path.name)

    # copy label theo tên ảnh (abc.jpg -> abc.txt)
    label_path = lbl_dir / (img_path.stem + ".txt")
    if label_path.exists():
        shutil.copy(label_path, out_lbl_dir / label_path.name)
        return True
    return False

for split in ["train", "valid", "test"]:
    src_img_dir = SRC / split / "images"
    src_lbl_dir = SRC / split / "labels"

    if not src_img_dir.exists():
        print(f"⚠ Skip {split}: không thấy {src_img_dir}")
        continue

    dst_img_dir = DST / split / "images"
    dst_lbl_dir = DST / split / "labels"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in src_img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    if not images:
        print(f"⚠ {split}: không có ảnh")
        continue

    random.shuffle(images)
    keep_count = max(1, int(len(images) * KEEP_RATIO))
    selected = images[:keep_count]

    missing = 0
    for img in selected:
        ok = copy_pair(img, src_lbl_dir, dst_img_dir, dst_lbl_dir)
        if not ok:
            missing += 1

    print(f"✔ {split}: copy {keep_count}/{len(images)} ảnh | thiếu label: {missing}")

print("✅ Done: dataset_split đã có images+labels theo đúng tên ảnh")
