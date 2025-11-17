"""
Hard Hat / PPE Detection - Training Script
==========================================
Command-line script for training YOLOv8 models on PPE detection dataset.
"""

import argparse
from pathlib import Path
import torch
from ultralytics import YOLO
import yaml


def get_batch_size(gpu_memory_gb):
    """Determine batch size based on GPU memory."""
    if gpu_memory_gb >= 12:
        return 16
    elif gpu_memory_gb >= 8:
        return 12
    elif gpu_memory_gb >= 6:
        return 8
    else:
        return 4


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 model for PPE detection')
    parser.add_argument('--data', type=str, default='data/data.yaml', help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolov8s.pt', help='Pretrained model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=None, help='Batch size (auto if not specified)')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu, auto if not specified)')
    parser.add_argument('--project', type=str, default='runs/detect', help='Project directory')
    parser.add_argument('--name', type=str, default='ppe_detection', help='Run name')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device is None:
        device = 0 if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Auto batch size
    if args.batch is None:
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            batch_size = get_batch_size(gpu_memory_gb)
            print(f"Auto-detected batch size: {batch_size} (GPU memory: {gpu_memory_gb:.2f} GB)")
        else:
            batch_size = 2
            print("Using CPU, batch size: 2")
    else:
        batch_size = args.batch
    
    # Load model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    # Load data config
    data_yaml = Path(args.data)
    if data_yaml.exists():
        with open(data_yaml, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Dataset: {config.get('names', [])}")
    
    # Train
    print(f"\nStarting training...")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Image size: {args.imgsz}")
    
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch_size,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        amp=True,
    )
    
    print(f"\n✅ Training complete!")
    print(f"Results saved to: {Path(model.trainer.save_dir) / 'weights' / 'best.pt'}")


if __name__ == '__main__':
    main()



