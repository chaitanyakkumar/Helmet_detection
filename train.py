"""
The file trains the model on helmet and non helmet classes
"""

from ultralytics import YOLO
import cv2, os, glob
import numpy as np
from PIL import Image
import IPython.display as display
import matplotlib.pyplot as plt

model = YOLO('yolov8s.pt')
data_yaml = '/content/drive/MyDrive/helmet_dataset/data.yaml'

model.train(
    data=data_yaml,
    epochs=30,
    imgsz=640,
    batch=16,
    lr0=0.001,
    device=0,  # GPU
    project='helmet_runs',   # wandb-safe
    name='helmet_yolov8s26',
    pretrained=True,
    workers=2,
    save_dir='/content/drive/MyDrive/helmet_runs'  # Drive output folder
)

# Load model
best_model_path = '/content/helmet_runs/helmet_yolov8s26/weights/best.pt'
model = YOLO(best_model_path)

# Validation images directory
val_dir = '/content/drive/MyDrive/helmet_dataset/images/val'
save_dir = '/content/infer_results/helmet_pred_custom'
os.makedirs(save_dir, exist_ok=True)

# Class names
names = model.names  # {0: 'helmet', 1: 'no_helmet'}

# Run inference
results = model.predict(source=val_dir, conf=0.5, device=0, verbose=False)

# Loop through results and draw bounding boxes
for i, r in enumerate(results):
    # Read original image
    img = r.orig_img.copy()

    for box in r.boxes:
        # Extract coordinates and class info
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls)
        conf = float(box.conf)
        label = f"{names[cls_id]} {conf:.2f}"

        # Color: green for helmet, red for no_helmet
        color = (0, 255, 0) if names[cls_id] == 'helmet' else (0, 0, 255)

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # Draw filled rectangle for label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - 25), (x1 + w, y1), color, -1)
        # Put label text
        cv2.putText(img, label, (x1, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save annotated image
    out_path = os.path.join(save_dir, f"pred_{i}.jpg")
    cv2.imwrite(out_path, img)

# Display first few predictions
pred_images = sorted(glob.glob(os.path.join(save_dir, '*.jpg')))[:5]
for img_path in pred_images:
    display.display(Image.open(img_path))

# Load best model
model = YOLO(best_model_path)

# Run evaluation on the validation set defined in data.yaml
metrics = model.val()

print(f"mAP@0.5: {metrics.box.map50*100:.2f}%")
print(f"Precision: {metrics.box.mp*100:.2f}%")
print(f"Recall: {metrics.box.mr*100:.2f}%")
print(f"mAP@0.5:0.95: {metrics.box.map*100:.2f}%")

# Load one 1080p frame (or video)
video_path = '/content/videoplayback (1).mp4'
cap = cv2.VideoCapture(video_path)

num_frames = 100
times = []

for i in range(num_frames):
    ret, frame = cap.read()
    if not ret:
        break
    start = time.time()
    results = model.predict(frame, imgsz=640, device=0, verbose=False)
    times.append(time.time() - start)

cap.release()
fps = 1 / np.mean(times)
print(f"Inference speed: {fps:.2f} FPS on 1080p video")

