import os, cv2, shutil, random
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
from utils import simplify_osf_label, to_yolo_bbox, write_label_file

root_dir = r"data"

# OSF
osf_ann_root   = os.path.join(root_dir, "OSF data", "annotation")
osf_img_root   = os.path.join(root_dir, "OSF data", "part_1")  # part_1..part_4

# Kaggle
kaggle_ann_root = os.path.join(root_dir, "kaggle data", "annotations")
kaggle_img_root = os.path.join(root_dir, "kaggle data", "images")

# Roboflow parent is the same folder that contains: roboflow_bike_helmet1/2/3
roboflow_parent = root_dir 

out_root = os.path.join(root_dir, "helmet_dataset")
split_ratio = 0.8
random.seed(42)

# Outputs
os.makedirs(out_root, exist_ok=True)
for sub in ["images/train","images/val","labels/train","labels/val"]:
    os.makedirs(os.path.join(out_root, sub), exist_ok=True)

# 1) OSF  
osf_records = []
print("Processing OSF data...")

for csv_file in os.listdir(osf_ann_root):
    if not csv_file.endswith(".csv"):
        continue

    csv_path = os.path.join(osf_ann_root, csv_file)
    folder_name = os.path.splitext(csv_file)[0]  # e.g., Bago_highway_1
    img_folder = os.path.join(osf_img_root, img_folder := folder_name)
    if not os.path.exists(img_folder):
        continue

    df = pd.read_csv(csv_path)
    prefix = f"osf_p1_{folder_name}"        # ✅ add a unique prefix for OSF-part1
    for _, row in df.iterrows():
        label = "no_helmet" if "NoHelmet" in str(row.label) else "helmet"
        frame_id = int(row.frame_id)
        img_name = f"{frame_id:02d}.jpg"
        img_path = os.path.join(img_folder, img_name)

        if not os.path.exists(img_path):
            continue

        # ✅ append 7-tuple including prefix
        osf_records.append((img_path, row.x, row.y, row.w, row.h, label, prefix))


# 2) Kaggle 
kaggle_records = []  # (img_path, xmin, ymin, w, h, label, prefix)
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

        fname = root.findtext("filename").strip()
        img_path = os.path.join(kaggle_img_root, fname)

        if not os.path.exists(img_path):
            base, _ = os.path.splitext(fname)
            for ext in [".jpg", ".jpeg", ".png"]:
                alt = os.path.join(kaggle_img_root, base + ext)
                if os.path.exists(alt):
                    img_path = alt
                    break
        if not os.path.exists(img_path):
            continue

        for obj in root.findall("object"):
            name = obj.findtext("name").strip().lower()
            label = "no_helmet" if ("without" in name or "no" in name) else "helmet"

            box = obj.find("bndbox")
            xmin, ymin = float(box.findtext("xmin")), float(box.findtext("ymin"))
            xmax, ymax = float(box.findtext("xmax")), float(box.findtext("ymax"))
            w, h = xmax - xmin, ymax - ymin
            kaggle_records.append((img_path, xmin, ymin, w, h, label, "kaggle"))

print(f"  → collected {len(kaggle_records):,} Kaggle boxes")

# 3) Roboflow (multiple datasets, each with train/valid/test)
print("Processing Roboflow data...")
rf_items = []  # (img_path, lbl_path, prefix)
rf_box_counts = {"helmet": 0, "no_helmet": 0}

def _gather_rf_set(ds_dir: str, ds_name: str):
    for split in ["train", "valid", "test"]:
        lbl_dir = os.path.join(ds_dir, split, "labels")
        img_dir = os.path.join(ds_dir, split, "images")
        if not (os.path.isdir(lbl_dir) and os.path.isdir(img_dir)):
            continue

        for lbl in os.listdir(lbl_dir):
            if not lbl.endswith(".txt") or lbl == "classes.txt":
                continue
            stem = os.path.splitext(lbl)[0]
            # match common image extensions
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                cand = os.path.join(img_dir, stem + ext)
                if os.path.exists(cand):
                    img_path = cand
                    break
            if not img_path:
                continue
            lbl_path = os.path.join(lbl_dir, lbl)
            rf_items.append((img_path, lbl_path, f"rf_{ds_name}"))

            # quick class count (assumes 0=helmet, 1=no_helmet like your data)
            try:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        cls_id = int(parts[0])
                        if cls_id == 0:
                            rf_box_counts["helmet"] += 1
                        else:
                            rf_box_counts["no_helmet"] += 1
            except Exception:
                pass

# find all roboflow_* datasets in the parent
for d in os.listdir(roboflow_parent):
    ds_dir = os.path.join(roboflow_parent, d)
    if not os.path.isdir(ds_dir):
        continue
    if not d.lower().startswith("roboflow_"):
        continue
    _gather_rf_set(ds_dir, d)

print(f"  → found {len(rf_items):,} Roboflow images "
      f"(boxes: helmet={rf_box_counts['helmet']}, no_helmet={rf_box_counts['no_helmet']})")

# Build a global image split
all_image_paths = set()
for t in osf_records:
    all_image_paths.add(t[0])
for t in kaggle_records:
    all_image_paths.add(t[0])
for img_path, _, _ in rf_items:
    all_image_paths.add(img_path)

all_image_paths = list(all_image_paths)
random.shuffle(all_image_paths)
train_cut = int(len(all_image_paths) * split_ratio)
img_to_split = {p: ("train" if i < train_cut else "val") for i, p in enumerate(all_image_paths)}

# Copy + label writing
stats = {'train_helmet': 0, 'train_no_helmet': 0,
         'val_helmet': 0, 'val_no_helmet': 0}

def ensure_image_copied(img_path: str, split: str, prefix: str) -> str:
    """Copy image once, return output filename used for label path."""
    base = os.path.basename(img_path)
    img_name = f"{prefix}_{base}"
    out_img = os.path.join(out_root, f"images/{split}/{img_name}")
    if not os.path.exists(out_img):
        shutil.copy(img_path, out_img)
    return img_name

# 3.1 OSF boxes
print("Copying OSF + Kaggle images and writing labels...")
for (img_path, x, y, w, h, label, prefix) in tqdm(osf_records, desc="OSF"):
    split = img_to_split.get(img_path, "train")
    img_name = ensure_image_copied(img_path, split, prefix)

    stem, _ = os.path.splitext(img_name)
    out_lbl = os.path.join(out_root, f"labels/{split}/{stem}.txt")

    img = cv2.imread(img_path)
    if img is None:
        continue
    ih, iw = img.shape[:2]
    bbox = to_yolo_bbox(x, y, w, h, iw, ih)
    cls_id = 0 if label == "helmet" else 1
    write_label_file(out_lbl, cls_id, bbox)
    stats[f"{split}_{label}"] += 1

# 3.2 Kaggle boxes
for (img_path, xmin, ymin, w, h, label, prefix) in tqdm(kaggle_records, desc="Kaggle"):
    split = img_to_split.get(img_path, "train")
    img_name = ensure_image_copied(img_path, split, prefix)

    stem, _ = os.path.splitext(img_name)
    out_lbl = os.path.join(out_root, f"labels/{split}/{stem}.txt")

    img = cv2.imread(img_path)
    if img is None:
        continue
    ih, iw = img.shape[:2]
    bbox = to_yolo_bbox(xmin, ymin, w, h, iw, ih)
    cls_id = 0 if label == "helmet" else 1
    write_label_file(out_lbl, cls_id, bbox)
    stats[f"{split}_{label}"] += 1

# 3.3 Roboflow (copy labels as-is, renamed to our stem)
print("Copying Roboflow images and labels...")
for (img_path, lbl_path, prefix) in tqdm(rf_items, desc="Roboflow"):
    split = img_to_split.get(img_path, "train")
    img_name = ensure_image_copied(img_path, split, prefix)

    stem, _ = os.path.splitext(img_name)
    out_lbl = os.path.join(out_root, f"labels/{split}/{stem}.txt")

    # append content (handles multi-obj)
    try:
        with open(lbl_path, "r", encoding="utf-8") as fsrc, open(out_lbl, "a", encoding="utf-8") as fdst:
            for line in fsrc:
                if not line.strip():
                    continue
                fdst.write(line)
                # quick stats
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    if cls_id == 0:
                        stats[f"{split}_helmet"] += 1
                    else:
                        stats[f"{split}_no_helmet"] += 1
    except Exception:
        continue

print("Dataset build complete!")
print(f"Train: Helmet={stats['train_helmet']}, NoHelmet={stats['train_no_helmet']}")
print(f"Val:   Helmet={stats['val_helmet']}, NoHelmet={stats['val_no_helmet']}")
print(f"Total unique images merged: {len(all_image_paths):,}")

# data.yaml
yaml_path = os.path.join(out_root, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"""path: {out_root.replace('\\','/')}
train: images/train
val: images/val
nc: 2
names: ['helmet', 'no_helmet']
""")
print(f"data.yaml written to {yaml_path}")
