import os
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 for Periapical Lesion Detection")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--weights", type=str, default="yolo11s.pt", help="Base weights")
    args = parser.parse_args()

    # Load a model
    print(f"Loading base model: {args.weights}")
    model = YOLO(args.weights)

    # Train the model
    print("Starting training...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project="runs/detect",
        name="periapical_yolo11s",
        patience=20, # Early stopping patience
        save=True,
        device="cuda" # Ensure it uses GPU on the workstation
    )
    print("Training completed. Results saved in runs/detect/periapical_yolo11s")

if __name__ == "__main__":
    main()
