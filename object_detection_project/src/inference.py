"""
Hard Hat / PPE Detection - Inference Script
============================================
Command-line script for running inference on images/videos using trained PPE detection model.
"""

import argparse
from pathlib import Path
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description='Run inference with PPE detection model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--source', type=str, required=True, help='Path to image/video or directory')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu, auto if not specified)')
    parser.add_argument('--save', action='store_true', help='Save results')
    parser.add_argument('--project', type=str, default='runs/detect', help='Project directory')
    parser.add_argument('--name', type=str, default='ppe_predictions', help='Run name')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device is None:
        device = 0 if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    print(f"Running inference on: {args.source}")
    print(f"  Device: {device}")
    print(f"  Confidence: {args.conf}")
    print(f"  IOU: {args.iou}")
    
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=device,
        save=args.save,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )
    
    print(f"\n✅ Inference complete!")
    if args.save:
        print(f"Results saved to: {Path(args.project) / args.name}")


if __name__ == '__main__':
    main()



