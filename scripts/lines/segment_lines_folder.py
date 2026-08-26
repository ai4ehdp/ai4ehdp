import cv2
from ultralytics import YOLO
import sys
import torch
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path

# -----------------------------
# Check command-line arguments
# -----------------------------
if len(sys.argv) < 4:
    print("Usage:\n python segment_lines_folder.py <model_lines> <input_dir> <output_dir>")
    sys.exit(1)

# -----------------------------
# Utility functions
# -----------------------------
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x

LINE_SEGMENTATION_IOU = 0.50
LINE_SEGMENTATION_CONF = 0.50
LINE_IMGSZ = 1024
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# -----------------------------
# Load YOLO model & set paths
# -----------------------------
model_for_lines = YOLO(sys.argv[1])
input_dir = Path(sys.argv[2])
output_dir = Path(sys.argv[3])

if not input_dir.exists() or not input_dir.is_dir():
    raise FileNotFoundError(f"Input directory not found: {input_dir}")

output_dir.mkdir(parents=True, exist_ok=True)

# Collect all image files
image_paths = [
    p for p in input_dir.iterdir()
    if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
]

if not image_paths:
    print(f"No supported images found in directory: {input_dir}")
    sys.exit(0)

print(f"Found {len(image_paths)} image(s) to process.\n")

# -----------------------------
# Process Images
# -----------------------------
for image_path in tqdm(image_paths, desc="Processing images"):
    orig_img = cv2.imread(str(image_path))
    if orig_img is None:
        print(f"Warning: Could not read {image_path.name}. Skipping.")
        continue

    img_h, img_w = orig_img.shape[:2]

    # Create dedicated crop folder per image (e.g., output_dir/line_crops/doc01/)
    img_stem = image_path.stem
    crop_dir = output_dir / "line_crops" / img_stem
    crop_dir.mkdir(parents=True, exist_ok=True)

    # Run YOLO prediction
    results_lines = model_for_lines.predict(
        source=str(image_path),
        verbose=False,
        imgsz=LINE_IMGSZ,
        iou=LINE_SEGMENTATION_IOU,
        conf=LINE_SEGMENTATION_CONF
    )
    res_lines = results_lines[0]

    if res_lines.boxes is None or len(res_lines.boxes) == 0:
        # Save original image to output_dir even if no lines detected
        cv2.imwrite(str(output_dir / image_path.name), orig_img)
        continue

    boxes_lines = to_numpy(res_lines.boxes.xyxy)
    confs_lines = to_numpy(res_lines.boxes.conf)

    # Sort lines vertically top-to-bottom by y1 coordinate
    sorted_indices = np.argsort(boxes_lines[:, 1])
    boxes_lines = boxes_lines[sorted_indices]
    confs_lines = confs_lines[sorted_indices]

    polygons_lines = None
    if hasattr(res_lines, "masks") and res_lines.masks is not None:
        polygons_lines = [res_lines.masks.xy[i] for i in sorted_indices]

    # Prepare translucent overlay
    overlay = orig_img.copy()
    color = (255, 255, 0)  # BGR Cyan
    alpha = 0.3            # Transparency factor

    # Crop and draw translucent fill
    for i, box in enumerate(boxes_lines):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        # Save line crop
        line_crop = orig_img[y1:y2, x1:x2]
        if line_crop.size > 0:
            cv2.imwrite(str(crop_dir / f"line_{i+1:03d}.png"), line_crop)

        # Draw fill on overlay
        if polygons_lines is not None and len(polygons_lines[i]) > 0:
            pts = polygons_lines[i].astype(np.int32)
            cv2.fillPoly(overlay, [pts], color=color)
        else:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    # Blend overlay with original image
    vis_img = cv2.addWeighted(overlay, alpha, orig_img, 1 - alpha, 0)

    # Draw solid boundary outlines & labels on blended image
    for i, box in enumerate(boxes_lines):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)

        if polygons_lines is not None and len(polygons_lines[i]) > 0:
            pts = polygons_lines[i].astype(np.int32)
            cv2.polylines(vis_img, [pts], isClosed=True, color=color, thickness=2)
        else:
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            vis_img, f"Line {i+1}", (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )

    # Save visual debug image to output_dir
    cv2.imwrite(str(output_dir / image_path.name), vis_img)

print("\nProcessing complete!")
print(f"- Visualizations saved to: '{output_dir}/'")
print(f"- Line crops saved to: '{output_dir / 'line_crops'}/'")