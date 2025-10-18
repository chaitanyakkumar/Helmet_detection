import os, cv2, shutil, random
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
from utils import simplify_osf_label, to_yolo_bbox, write_label_file

# Setup directories
root_dir = r"D:\Capstone project\data"
osf_ann_root = os.path.join(root_dir, "OSF data", "annotation")
osf_img_root = os.path.join(root_dir, "OSF data", "part_1")
kaggle_ann_root = os.path.join(root_dir, "kaggle data", "annotations")
kaggle_img_root = os.path.join(root_dir, "kaggle data", "images")
out_root = os.path.join(root_dir, "helmet_dataset")

split_ratio = 0.8   #split the data to 80 and 20%    

os.makedirs(out_root, exist_ok=True)
for sub in ["images/train","images/val","labels/train","labels/val"]:
    os.makedirs(os.path.join(out_root, sub), exist_ok=True)


# Preprocess the OSF data
osf_records = []
print("Processing OSF data...")

for csv_file in os.listdir(osf_ann_root):
    if not csv_file.endswith(".csv"):
        continue

    csv_path = os.path.join(osf_ann_root, csv_file)
    folder_name = os.path.splitext(csv_file)[0]  # e.g., Bago_highway_1
    img_folder = os.path.join(osf_img_root, folder_name)
    if not os.path.exists(img_folder):
        continue

    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        label = "no_helmet" if "NoHelmet" in str(row.label) else "helmet"
        frame_id = int(row.frame_id)
        # Format frame_id as two digits, e.g., 1 → 01.jpg
        img_name = f"{frame_id:02d}.jpg"
        img_path = os.path.join(img_folder, img_name)

        if not os.path.exists(img_path):
            continue

        osf_records.append((img_path, row.x, row.y, row.w, row.h, label))

print(f"  → collected {len(osf_records):,} OSF boxes")

# Preprocess the kaggle data
kaggle_records = []
print("Processing Kaggle data...")

for xml_file in os.listdir(kaggle_ann_root):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(kaggle_ann_root, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    fname = root.findtext("filename").strip()
    img_path = os.path.join(kaggle_img_root, fname)

    # Handle .png/.jpg mismatches
    if not os.path.exists(img_path):
        base, _ = os.path.splitext(fname)
        for ext in [".jpg", ".jpeg", ".png"]:
            alt = os.path.join(kaggle_img_root, base + ext)
            if os.path.exists(alt):
                img_path = alt
                break
    if not os.path.exists(img_path):
        continue

    img_w = int(root.find("size/width").text)
    img_h = int(root.find("size/height").text)

    for obj in root.findall("object"):
        name = obj.findtext("name").strip().lower()
        if "without" in name or "no" in name:
            label = "no_helmet"
        else:
            label = "helmet"

        box = obj.find("bndbox")
        xmin, ymin = float(box.findtext("xmin")), float(box.findtext("ymin"))
        xmax, ymax = float(box.findtext("xmax")), float(box.findtext("ymax"))
        w, h = xmax - xmin, ymax - ymin
        kaggle_records.append((img_path, xmin, ymin, w, h, label))

print(f"  → collected {len(kaggle_records):,} Kaggle boxes")

# Merge both data sets and create train and Val sets
records = osf_records + kaggle_records
random.shuffle(records)

train_count = int(len(records) * split_ratio)
train_set = set(range(train_count))

# Counters
stats = {'train_helmet': 0, 'train_no_helmet': 0,
         'val_helmet': 0, 'val_no_helmet': 0}

for i, rec in enumerate(tqdm(records, desc="Copying & converting")):
    img_path, x, y, w, h, label = rec
    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    img_h, img_w = img.shape[:2]
    bbox = to_yolo_bbox(x, y, w, h, img_w, img_h)
    cls_id = 0 if label == "helmet" else 1
    split = "train" if i in train_set else "val"

    # -------------------------------
    # Fix: unique naming for OSF only
    # -------------------------------
    subfolder = os.path.basename(os.path.dirname(img_path))
    base_name = os.path.basename(img_path)

    # Prefix only if from OSF (Bago_highway_X folders)
    if "Bago_highway" in subfolder:
        img_name = f"{subfolder}_{base_name}"
    else:
        img_name = base_name

    base, _ = os.path.splitext(img_name)

    out_img = os.path.join(out_root, f"images/{split}/{img_name}")
    out_lbl = os.path.join(out_root, f"labels/{split}/{base}.txt")

    # Copy image
    if not os.path.exists(out_img):
        shutil.copy(img_path, out_img)

    # Append box line to label file (multiple objects per frame)
    write_label_file(out_lbl, cls_id, bbox)

    # Update stats
    key = f"{split}_{label}"
    stats[key] += 1

print("✅ Dataset build complete!")
print(f"Train: Helmet={stats['train_helmet']}, NoHelmet={stats['train_no_helmet']}")
print(f"Val:   Helmet={stats['val_helmet']}, NoHelmet={stats['val_no_helmet']}")
print(f"Total: {len(records)} images (approx; some may have multiple boxes)")

# Create the YAML file
yaml_path = os.path.join(out_root, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"""path: {out_root.replace('\\','/')}
train: images/train
val: images/val
nc: 2
names: ['helmet', 'no_helmet']
""")
print(f"data.yaml written to {yaml_path}")


