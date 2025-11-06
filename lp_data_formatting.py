import os, cv2, shutil, random, glob, re
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
from utils import to_yolo_bbox, write_label_file  # simplify_osf_label not needed here

# CONFIG
root_dir = r"data"

# OSF (using part_1 only, as requested)
osf_ann_root = os.path.join(root_dir, "OSF data", "annotation")
osf_img_root = os.path.join(root_dir, "OSF data", "part_1")

# Kaggle
kaggle_ann_root = os.path.join(root_dir, "kaggle data", "annotations")
kaggle_img_root = os.path.join(root_dir, "kaggle data", "images")

# Roboflow datasets (YOLOv8)
rf_helmet_dirs = [
    os.path.join(root_dir, "roboflow_bike_helmet1"),
    os.path.join(root_dir, "roboflow_bike_helmet2"),
    os.path.join(root_dir, "roboflow_bike_helmet3"),
]
rf_lp_dirs = [
    os.path.join(root_dir, "license_plate1"),
    os.path.join(root_dir, "license_plate2"),
    os.path.join(root_dir, "license_plate3"),
]

# Output
out_root = os.path.join(root_dir, "helmet_dataset")
split_ratio = 0.8  # global split for *all* images
random.seed(42)
IMG_EXTS = [".jpg", ".jpeg", ".png"]

# SETUP OUTPUT
os.makedirs(out_root, exist_ok=True)
for sub in ["images/train","images/val","labels/train","labels/val"]:
    os.makedirs(os.path.join(out_root, sub), exist_ok=True)

# HELPERS
def find_img_with_stem(img_dir: str, stem: str):
    for ext in IMG_EXTS:
        p = os.path.join(img_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None

def ensure_image_copied(img_path: str, split: str, prefix: str) -> str:
    base = os.path.basename(img_path)
    img_name = f"{prefix}_{base}"
    out_img = os.path.join(out_root, f"images/{split}/{img_name}")
    if not os.path.exists(out_img):
        shutil.copy(img_path, out_img)
    return img_name

def read_names_from_data_yaml(dataset_dir: str):
    """Read names list from <dataset>/data.yaml (simple parser, no PyYAML)."""
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    names = None
    if os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            txt = f.read()
        # naive parse for a line like: names: ['helmet','no_helmet']
        m = re.search(r"names\s*:\s*\[([^\]]+)\]", txt)
        if m:
            raw = m.group(1)
            # split by commas, strip quotes/spaces
            names = [re.sub(r"['\"\s]", "", s) for s in raw.split(",")]
    return names  # may be None

# 1) OSF 
osf_records = []  # (img_path, x, y, w, h, cls_id(0/1), prefix)
print("Processing OSF data...")

if os.path.isdir(osf_ann_root):
    for csv_file in os.listdir(osf_ann_root):
        if not csv_file.endswith(".csv"):
            continue

        csv_path = os.path.join(osf_ann_root, csv_file)
        folder_name = os.path.splitext(csv_file)[0]  # e.g., Bago_highway_1
        img_folder = os.path.join(osf_img_root, folder_name)
        if not os.path.exists(img_folder):
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        prefix = f"osf_p1_{folder_name}"
        # Expect columns: frame_id, x, y, w, h, label (Helmet/NoHelmet)
        for _, row in df.iterrows():
            try:
                frame_id = int(row.frame_id)
                x, y, w, h = float(row.x), float(row.y), float(row.w), float(row.h)
            except Exception:
                continue
            lab = str(row.label)
            cls_id = 1 if ("NoHelmet" in lab) else 0  # 0=helmet, 1=no_helmet

            img_name = f"{frame_id:02d}.jpg"
            img_path = os.path.join(img_folder, img_name)
            if not os.path.exists(img_path):
                continue

            osf_records.append((img_path, x, y, w, h, cls_id, prefix))

print(f"  → OSF boxes: {len(osf_records):,}")

# 2) Kaggle (Pascal VOC XML)
kaggle_records = []  # (img_path, x, y, w, h, cls_id(0/1), prefix)
print("Processing Kaggle data...")

if os.path.isdir(kaggle_ann_root):
    for xml_file in os.listdir(kaggle_ann_root):
        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(kaggle_ann_root, xml_file)
        try:
            tree = ET.parse(xml_path)
        except Exception:
            continue
        root = tree.getroot()

        fname = (root.findtext("filename") or "").strip()
        img_path = os.path.join(kaggle_img_root, fname)
        if not os.path.exists(img_path):
            base, _ = os.path.splitext(fname)
            img_path = find_img_with_stem(kaggle_img_root, base)
        if not img_path:
            continue

        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip().lower()
            cls_id = 1 if ("without" in name or "no" in name) else 0
            box = obj.find("bndbox")
            xmin = float(box.findtext("xmin")); ymin = float(box.findtext("ymin"))
            xmax = float(box.findtext("xmax")); ymax = float(box.findtext("ymax"))
            w = xmax - xmin; h = ymax - ymin
            kaggle_records.append((img_path, xmin, ymin, w, h, cls_id, "kaggle"))

print(f"  → Kaggle boxes: {len(kaggle_records):,}")

# 3) Roboflow (YOLOv8)
#     - Helmet sets: remap according to their data.yaml names to (0,1)
#     - License plate sets: remap any class to 2
print("Processing Roboflow helmet datasets...")
rf_helmet_items = []  # (img_path, lbl_path, prefix, remap_dict)
for ds_dir in rf_helmet_dirs:
    if not os.path.isdir(ds_dir):
        continue
    names = read_names_from_data_yaml(ds_dir) or ['helmet', 'no_helmet']
    # Build mapping from dataset's class index -> our id (0/1)
    remap = {}
    for i, n in enumerate(names):
        n_clean = n.strip().lower()
        if n_clean in ("helmet",):
            remap[i] = 0
        elif n_clean in ("no_helmet", "nohelmet", "without_helmet", "withouthelmet"):
            remap[i] = 1
    for split in ["train", "valid", "test"]:
        lbl_dir = os.path.join(ds_dir, split, "labels")
        img_dir = os.path.join(ds_dir, split, "images")
        if not (os.path.isdir(lbl_dir) and os.path.isdir(img_dir)):
            continue
        for lbl in os.listdir(lbl_dir):
            if not lbl.endswith(".txt") or lbl == "classes.txt":
                continue
            stem = os.path.splitext(lbl)[0]
            img_path = find_img_with_stem(img_dir, stem)
            if not img_path:
                continue
            rf_helmet_items.append((img_path, os.path.join(lbl_dir, lbl), f"rf_{os.path.basename(ds_dir)}", remap))

print(f"  → Roboflow helmet images: {len(rf_helmet_items):,}")

print("Processing Roboflow license-plate datasets...")
rf_lp_items = []  # (img_path, lbl_path, prefix)
for ds_dir in rf_lp_dirs:
    if not os.path.isdir(ds_dir):
        continue
    for split in ["train", "valid", "test"]:
        lbl_dir = os.path.join(ds_dir, split, "labels")
        img_dir = os.path.join(ds_dir, split, "images")
        if not (os.path.isdir(lbl_dir) and os.path.isdir(img_dir)):
            continue
        for lbl in os.listdir(lbl_dir):
            if not lbl.endswith(".txt") or lbl == "classes.txt":
                continue
            stem = os.path.splitext(lbl)[0]
            img_path = find_img_with_stem(img_dir, stem)
            if not img_path:
                continue
            rf_lp_items.append((img_path, os.path.join(lbl_dir, lbl), f"rf_{os.path.basename(ds_dir)}"))

print(f"  → Roboflow license-plate images: {len(rf_lp_items):,}")

# 4) Build global split
all_imgs = set()
for t in osf_records: all_imgs.add(t[0])
for t in kaggle_records: all_imgs.add(t[0])
for img_path, *_ in rf_helmet_items: all_imgs.add(img_path)
for img_path, *_ in rf_lp_items: all_imgs.add(img_path)

all_imgs = list(all_imgs)
random.shuffle(all_imgs)
train_cut = int(len(all_imgs) * split_ratio)
img_to_split = {p: ("train" if i < train_cut else "val") for i, p in enumerate(all_imgs)}

# 5) Write images + labels
stats = {
    'train_helmet': 0, 'train_no_helmet': 0, 'train_license_plate': 0,
    'val_helmet':   0, 'val_no_helmet':   0, 'val_license_plate':   0
}

print("Copying images and writing labels...")

# 5.1 OSF
for (img_path, x, y, w, h, cls_id, prefix) in tqdm(osf_records, desc="OSF"):
    split = img_to_split.get(img_path, "train")
    img_name = ensure_image_copied(img_path, split, prefix)
    stem, _ = os.path.splitext(img_name)
    out_lbl = os.path.join(out_root, f"labels/{split}/{stem}.txt")
    img = cv2.imread(img_path)
    if img is None: 
        continue
    ih, iw = img.shape[:2]
    bbox = to_yolo_bbox(x, y, w, h, iw, ih)
    write_label_file(out_lbl, cls_id, bbox)
    key = ('train_' if split=='train' else 'val_') + ('helmet' if cls_id==0 else 'no_helmet')
    stats[key] += 1

# 5.2 Kaggle
for (img_path, x, y, w, h, cls_id, prefix) in tqdm(kaggle_records, desc="Kaggle"):
    split = img_to_split.get(img_path, "train")
    img_name = ensure_image_copied(img_path, split, prefix)
    stem, _ = os.path.splitext(img_name)
    out_lbl = os.path.join(out_root, f"labels/{split}/{stem}.txt")
    img = cv2.imread(img_path)
    if img is None: 
        continue
    ih, iw = img.shape[:2]
    bbox = to_yolo_bbox(x, y, w, h, iw, ih)
    write_label_file(out_lbl, cls_id, bbox)
    key = ('train_' if split=='train' else 'val_') + ('helmet' if cls_id==0 else 'no_helmet')
    stats[key] += 1

# 5.3 Roboflow helmet (remap to 0/1 using per-dataset names)
for (img_path, lbl_path, prefix, remap) in tqdm(rf_helmet_items, desc="RF-Helmet"):
    split = img_to_split.get(img_path, "train")
    # Send dataset "test" to val by just honoring img_to_split (built globally)
    img_name = ensure_image_copied(img_path, split, prefix)
    stem, _ = os.path.splitext(img_name)
    out_lbl = os.path.join(out_root, f"labels/{split}/{stem}.txt")
    try:
        with open(lbl_path, "r", encoding="utf-8") as fsrc, open(out_lbl, "a", encoding="utf-8") as fdst:
            for line in fsrc:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                old_id = int(parts[0])
                if old_id not in remap:
                    # skip any class we don't recognize
                    continue
                parts[0] = str(remap[old_id])  # 0 or 1
                fdst.write(" ".join(parts) + "\n")
                cls_id = remap[old_id]
                key = ('train_' if split=='train' else 'val_') + ('helmet' if cls_id==0 else 'no_helmet')
                stats[key] += 1
    except Exception:
        continue

# 5.4 Roboflow license-plate (force id=2)
for (img_path, lbl_path, prefix) in tqdm(rf_lp_items, desc="RF-LicensePlate"):
    split = img_to_split.get(img_path, "train")
    img_name = ensure_image_copied(img_path, split, prefix)
    stem, _ = os.path.splitext(img_name)
    out_lbl = os.path.join(out_root, f"labels/{split}/{stem}.txt")
    try:
        with open(lbl_path, "r", encoding="utf-8") as fsrc, open(out_lbl, "a", encoding="utf-8") as fdst:
            for line in fsrc:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                parts[0] = "2"
                fdst.write(" ".join(parts) + "\n")
                key = ('train_' if split=='train' else 'val_') + 'license_plate'
                stats[key] += 1
    except Exception:
        continue

print("Dataset build complete!")
print(f"Train: Helmet={stats['train_helmet']}, NoHelmet={stats['train_no_helmet']}, LicensePlate={stats['train_license_plate']}")
print(f"Val:   Helmet={stats['val_helmet']},   NoHelmet={stats['val_no_helmet']},   LicensePlate={stats['val_license_plate']}")

# Write data.yaml (3 classes)
yaml_path = os.path.join(out_root, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"""path: {out_root.replace('\\','/')}
train: images/train
val: images/val
nc: 3
names: ['helmet', 'no_helmet', 'license_plate']
""")
print(f"data.yaml written to {yaml_path}")
