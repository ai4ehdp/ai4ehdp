import cv2
from ultralytics import YOLO
import sys
import torch
import numpy as np
from shapely.geometry import Polygon
import os
from tqdm import tqdm
from pathlib import Path

# -----------------------------
# Check command-line arguments
# -----------------------------
if len(sys.argv) < 4:
    print("Usage:\n python segment_lines.py <model_lines> <image_path> <output_img_path>")
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

# -----------------------------
# Load YOLO model & set paths
# -----------------------------
model_for_lines = YOLO(sys.argv[1])
image_path = sys.argv[2]
output_path = sys.argv[3]

# Ensure output folders exist
crop_dir = Path('line_crops')
crop_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Read image
# -----------------------------
orig_img = cv2.imread(image_path)
if orig_img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")
img_h, img_w = orig_img.shape[:2]

# -----------------------------
# Segment lines in the image
# -----------------------------
results_lines = model_for_lines.predict(
    source=image_path,
    verbose=False,
    imgsz=LINE_IMGSZ,
    iou=LINE_SEGMENTATION_IOU,
    conf=LINE_SEGMENTATION_CONF
)
res_lines = results_lines[0]

if res_lines.boxes is None or len(res_lines.boxes) == 0:
    print("No line segments detected.")
    sys.exit(0)

boxes_lines = to_numpy(res_lines.boxes.xyxy)
confs_lines = to_numpy(res_lines.boxes.conf)

# Sort lines vertically top-to-bottom by y1 coordinate
sorted_indices = np.argsort(boxes_lines[:, 1])
boxes_lines = boxes_lines[sorted_indices]
confs_lines = confs_lines[sorted_indices]

polygons_lines = None
if hasattr(res_lines, "masks") and res_lines.masks is not None:
    polygons_lines = [res_lines.masks.xy[i] for i in sorted_indices]

# -----------------------------
# Process & Crop Lines
# -----------------------------
overlay = orig_img.copy()
color = (255, 255, 0)  # BGR Cyan
alpha = 0.3  # Fill transparency level (0.0 to 1.0)

for i, box in enumerate(tqdm(boxes_lines, desc="Processing line crops")):
    x1, y1, x2, y2 = map(int, box)

    # Clamp coordinates within image boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)

    # Save bounding box crop
    line_crop = orig_img[y1:y2, x1:x2]
    if line_crop.size > 0:
        cv2.imwrite(str(crop_dir / f"line_{i+1:03d}.png"), line_crop)

    # Draw filled translucent shape onto overlay
    if polygons_lines is not None and len(polygons_lines[i]) > 0:
        pts = polygons_lines[i].astype(np.int32)
        cv2.fillPoly(overlay, [pts], color=color)
    else:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

# Blend filled overlay with original image
vis_img = cv2.addWeighted(overlay, alpha, orig_img, 1 - alpha, 0)

# -----------------------------
# Draw Crisp Outlines & Labels
# -----------------------------
for i, box in enumerate(boxes_lines):
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)

    # Draw solid boundary outline
    if polygons_lines is not None and len(polygons_lines[i]) > 0:
        pts = polygons_lines[i].astype(np.int32)
        cv2.polylines(vis_img, [pts], isClosed=True, color=color, thickness=2)
    else:
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

    # Label line index
    cv2.putText(
        vis_img, f"Line {i+1}", (x1, max(15, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
    )

# -----------------------------
# Save final visual debug image
# -----------------------------
cv2.imwrite(output_path, vis_img)
print(f"Successfully processed {len(boxes_lines)} lines.")
print(f"Crops saved to: '{crop_dir}/'")
print(f"Visualization saved to: '{output_path}'")