# AI for Road Safety

Computer vision system for detecting **helmet usage** and **license plates** from traffic images and videos using YOLOv8 and lightweight OCR.

---

## 🔍 Overview

This project builds an end-to-end pipeline for **road safety monitoring**:

- Detects:
  - `helmet`
  - `no_helmet`
  - `license_plate`
- Works on:
  - Static images (validation / test sets)
  - Video files and camera streams
- For detected **license plates**, it:
  - Crops the plate region
  - Runs basic OCR (OpenCV + Tesseract)
  - Overlays the recognized plate number on the frame

The focus is on **practical, deployable AI**: efficient inference, configurable training, and interpretable outputs for downstream analytics (e.g., helmet compliance statistics, violation alerts).

---

##  Key Features

- **YOLOv8-based object detection**
  - 3 classes: `helmet`, `no_helmet`, `license_plate`
  - Multiple training strategies:
    - Layer freezing (`freeze=10`, `freeze=5`)
    - Custom fine-tuning: **unfreeze detection head + last 10 backbone layers**
  - Cosine learning-rate schedule for smoother convergence

- **Data processing utilities**
  - Scripts for data cleaning and augmentation
  - Support for combining datasets from multiple sources

- **Image & video inference**
  - Batch inference on image folders with annotated outputs
  - Video inference with **frame sampling** (e.g. from 24 FPS → ~3 FPS) to reduce compute while preserving temporal coverage
  - Color-coded boxes:
    - Green: helmet
    - Red: no helmet
    - Yellow/Cyan: license plate

- **License plate OCR (only when needed)**
  - OCR is run **only** on detections with class `license_plate`
  - OpenCV pre-processing + Tesseract OCR
  - Basic cleaning and validation of recognized text (A–Z, 0–9)

---

##  Data Sources

We aggregate helmet and traffic datasets from multiple public sources:

- **Kaggle** – helmet / non-helmet rider datasets
- **Roboflow** – curated and annotated traffic / helmet / license-plate datasets
- **OSF (Open Science Framework)** – research datasets for traffic surveillance

Each source is preprocessed and normalized into a **YOLO-compatible format**:
- Images: `images/{train,val,test}`
- Labels: `labels/{train,val,test}` (YOLO TXT format)
- Dataset configuration is defined in `data.yaml` with:

```yaml
names:
  0: helmet
  1: no_helmet
  2: license_plate
```


## Project Structure
```
.
├── data/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/           # optional
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/           # optional
│   └── data.yaml           # YOLO dataset configuration
├── train.py                # Training experiments (freezing / unfreezing)
├── inference.py            # Image + video inference + OCR for license plates
├── data_cleaning.py        # Cleaning, sanity checks, split utilities
├── data_augmentation.py    # Augmentation and visualization helpers
├── lp_data_formatting.py   # (Optional) tools to standardize license-plate labels
├── utils.py                # Shared helper functions
└── README.md               # Project documentation
```

## Training 

- `train.py` uses `data/data.yaml` with 3 classes: `helmet`, `no_helmet`, `license_plate`.  
- Loads YOLOv8 base weights and runs multiple experiments with different freezing strategies.  
- Includes a custom setup that unfreezes only the detection head and last 10 backbone layers.  
- Uses a cosine learning-rate schedule and saves `best.pt` for each run under `helmet_runs/`.  
- Evaluates mAP, precision, recall on the validation set and can benchmark FPS on a sample video.  

## Inference

- `inference.py` loads the trained `best.pt` model and supports both image and video inputs.  
- For images, it annotates detections with color-coded boxes (green: helmet, red: no_helmet, yellow: license_plate).  
- For license-plate detections only, it crops the region, runs OpenCV + Tesseract OCR, and overlays the plate text.  
- For videos, it samples frames (e.g., 24 FPS → ~3 FPS), runs detection + OCR, and writes an annotated output video.  
- Annotated images and sampled frames are saved to output folders for visualization and further analysis.  
