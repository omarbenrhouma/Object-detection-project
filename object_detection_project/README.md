# Hard Hat / PPE Detection System using YOLO and GPU Acceleration

A computer vision project implementing real-time Hard Hat / Personal Protective Equipment (PPE) detection with YOLO. The system leverages NVIDIA GPU acceleration for optimal performance and can detect workers wearing hard hats, heads without helmets, and persons in workplace settings.

## Project Overview

This project implements an end-to-end object detection system for workplace safety monitoring. The model detects three classes:
- **Head**: Worker's head without helmet (safety violation)
- **Helmet**: Worker wearing hard hat (compliant)
- **Person**: General person detection

The system is designed for real-time monitoring of construction sites, factories, and other workplaces where hard hat compliance is required.

## Features

- ✅ CUDA-enabled GPU acceleration
- ✅ Fine-tuned YOLOv8 model for PPE detection
- ✅ Three-class detection (head, helmet, person)
- ✅ Real-time inference capability
- ✅ Comprehensive data preprocessing and augmentation
- ✅ Two-phase training pipeline (frozen backbone + full fine-tuning)
- ✅ GPU-optimized batch size adjustment

## Project Structure

```
object_detection_project/
├── data/                   # Dataset (train/valid/test splits)
│   ├── data.yaml          # Dataset configuration
│   ├── train/             # Training images and labels
│   ├── valid/             # Validation images and labels
│   └── test/              # Test images and labels
├── notebooks/              # Jupyter notebooks
│   ├── 01_data_collection.ipynb    # Dataset validation
│   ├── 02_eda.ipynb                # Exploratory data analysis
│   ├── 03_train_ppe.ipynb          # Model training
│   └── 04_predict_ppe.ipynb         # Inference and predictions
├── models/                 # Model weights (gitignored)
├── runs/                   # Training outputs (gitignored)
├── src/                    # Source code
│   ├── train.py           # Training script
│   ├── inference.py       # Inference script
│   └── utils.py           # Utility functions
├── deployment/             # Deployment scripts
│   └── app.py             # Flask deployment app
└── README.md              # This file
```

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (tested on RTX 3060)
- Minimum 6GB GPU memory
- CUDA 12.6+

### Software
- Python 3.8+
- PyTorch with CUDA support
- Ultralytics YOLO
- Label Studio (for annotation)

## Installation

### 1. CUDA Setup
1. Download and install [CUDA Toolkit 12.6](https://developer.nvidia.com/cuda-downloads)
2. Download and install [cuDNN](https://developer.nvidia.com/cudnn)
3. Add CUDA to system PATH:
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin
   ```

### 2. Python Environment
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install dependencies
pip install ultralytics label-studio numpy pandas matplotlib pillow
```

### 3. Verify GPU
```python
import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))
```

## Usage

### 1. Data Preparation
The dataset should be organized in YOLO format:
```
data/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

The `data.yaml` file should contain:
```yaml
train: train/images
val: valid/images
test: test/images
nc: 3
names: ['head', 'helmet', 'person']
```

### 2. Dataset Validation
```bash
jupyter notebook notebooks/01_data_collection.ipynb
```
This notebook validates the dataset structure and provides statistics.

### 3. Exploratory Data Analysis
```bash
jupyter notebook notebooks/02_eda.ipynb
```
Analyzes class distribution, image sizes, bounding box statistics, and visualizes samples.

### 4. Training
```bash
jupyter notebook notebooks/03_train_ppe.ipynb
```
The notebook will:
- Automatically detect GPU and adjust batch size
- Perform two-phase training (frozen backbone + full fine-tuning)
- Save best model to `runs/detect/ppe_detection_phase2/weights/best.pt`

Alternatively, use the command-line script:
```bash
python src/train.py --data data/data.yaml --model yolov8s.pt --epochs 30
```

### 5. Prediction
```bash
jupyter notebook notebooks/04_predict_ppe.ipynb
```
This notebook:
- Loads the trained model
- Runs predictions on test set
- Visualizes results with color-coded bounding boxes
- Provides statistics and analysis

Or use the command-line script:
```bash
python src/inference.py --model runs/detect/ppe_detection_phase2/weights/best.pt --source data/test/images --save
```

## Deployment

### Flask Web App
Run the deployment app:
```bash
cd deployment
python app.py
```

Then open `http://localhost:5000` in your browser to upload images and get predictions.

## Results

After training, check the validation metrics in the training notebook. Typical results include:
- **mAP@0.5**: Mean Average Precision at IoU 0.5
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds
- **Precision**: Percentage of correct positive predictions
- **Recall**: Percentage of actual positives detected

### Inference Speed
- **GPU**: ~20-50ms per image (depending on GPU)
- **Real-time**: ~20-50 FPS on RTX 3060
- **Batch processing**: Supported

## Dataset

This project uses the Hard Hat Workers dataset from Roboflow:
- **Source**: [Roboflow Universe](https://universe.roboflow.com/car-4lgko/hard-hat-workers-beiaj)
- **License**: Public Domain
- **Classes**: head, helmet, person
- **Splits**: Train/Valid/Test (70/20/10)

## License

This project is for educational purposes.

## Authors
- Omar Mejri
- Omar Ben Rhouma


## Acknowledgments

Special thanks to the professor for guidance and support throughout the project.
