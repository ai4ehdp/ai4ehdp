import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(
        description="Train a YOLO classification model on a historical document dataset."
    )
    
    # Required argument for dataset path
    parser.add_argument(
        "-d", "--data",
        type=str,
        required=True,
        help="Path to the dataset folder (e.g., dataset_full or dataset_quater)"
    )
    
    # Optional training configuration arguments
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="yolo11m-cls.pt",
        help="Base model checkpoint (default: yolo11m-cls.pt)"
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="Input image size (default: 224)"
    )
    parser.add_argument(
        "-b", "--batch",
        type=int,
        default=8,
        help="Batch size (default: 8)"
    )

    args = parser.parse_args()

    # Load model architecture
    model = YOLO(args.model)

    # Train model
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch
    )

if __name__ == "__main__":
    main()