import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO model classification inference on historical document images."
    )
    
    # Required command line argument for the model path
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="Path to the YOLO model file (e.g., models/full/best.pt or models/quater/best.pt)"
    )
    
    # Optional command line argument for the input source directory
    parser.add_argument(
        "-s", "--source",
        type=str,
        default=".",
        help="Path to folder or image file for inference"
    )

    args = parser.parse_args()

    # Load model from command line argument
    model = YOLO(args.model)

    # Run prediction
    model.predict(
        source=args.source,
        save=True,       # Saves annotated images to runs/predict/
        show_conf=True   # Displays confidence score
    )

if __name__ == "__main__":
    main()