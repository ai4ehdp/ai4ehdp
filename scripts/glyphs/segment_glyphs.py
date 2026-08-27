import cv2
from ultralytics import YOLO
import sys
import torch
import numpy as np
import supervision as sv
from tqdm import tqdm
from pathlib import Path

# -----------------------------
# Check command-line arguments
# -----------------------------
if len(sys.argv) < 4:
    print("Usage:\n python segment_glyphs.py <model_glyphs> <image_path> <output_img_path> [--slicing]")
    sys.exit(1)

# -----------------------------
# Utility functions
# -----------------------------
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x

def masks_to_polygons(masks):
    polygons = []
    if masks is None:
        return None

    for mask in masks:
        mask_uint8 = mask.astype(bool).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            polygons.append(largest_contour.reshape(-1, 2))
        else:
            polygons.append(np.empty((0, 2), dtype=np.int32))

    return polygons

def predict_glyphs_with_slicing(model, image):
    def callback(image_slice):
        results = model.predict(
            source=image_slice,
            verbose=False,
            imgsz=GLYPH_IMGSZ,
            iou=GLYPH_SEGMENTATION_IOU,
            conf=GLYPH_SEGMENTATION_CONF
        )
        return sv.Detections.from_ultralytics(results[0])

    if isinstance(GLYPHS_SLICER_SLICE_WH, int):
        slice_w, slice_h = GLYPHS_SLICER_SLICE_WH, GLYPHS_SLICER_SLICE_WH
    else:
        slice_w, slice_h = GLYPHS_SLICER_SLICE_WH

    if isinstance(GLYPH_SLICER_OVERLAP_RATIO_WH, (int, float)):
        ratio_w, ratio_h = GLYPH_SLICER_OVERLAP_RATIO_WH, GLYPH_SLICER_OVERLAP_RATIO_WH
    else:
        ratio_w, ratio_h = GLYPH_SLICER_OVERLAP_RATIO_WH

    slicer_kwargs = {
        "callback": callback,
        "slice_wh": GLYPHS_SLICER_SLICE_WH,
        "overlap_wh": (int(slice_w * ratio_w), int(slice_h * ratio_h)),
        "iou_threshold": GLYPH_SEGMENTATION_IOU,
        "thread_workers": GLYPH_SLICER_THREAD_WORKERS,
    }

    slicer = sv.InferenceSlicer(**slicer_kwargs)
    return slicer(image)

GLYPH_SEGMENTATION_IOU = 0.50
GLYPH_SEGMENTATION_CONF = 0.50
GLYPH_IMGSZ = 640

USE_GLYPH_SLICER = "--slicing" in sys.argv
GLYPHS_SLICER_SLICE_WH = (640, 640)
GLYPH_SLICER_OVERLAP_RATIO_WH = (0.15, 0.15)
GLYPH_SLICER_THREAD_WORKERS = 2

# -----------------------------
# Load YOLO model & set paths
# -----------------------------
model_for_glyphs = YOLO(sys.argv[1])
image_path = sys.argv[2]
output_path = sys.argv[3]
class_names = model_for_glyphs.names
class_count = len(class_names)
class_colors = []
for cls_id in range(class_count):
    hue = (cls_id * 37) % 180
    color_hsv = np.uint8([[[hue, 220, 255]]])
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    class_colors.append(tuple(map(int, color_bgr)))

# Ensure output folders exist
crop_dir = Path('glyph_crops')
crop_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Read image
# -----------------------------
orig_img = cv2.imread(image_path)
if orig_img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")
img_h, img_w = orig_img.shape[:2]

# -----------------------------
# Segment glyphs in the image
# -----------------------------
if USE_GLYPH_SLICER:
    detections_glyphs = predict_glyphs_with_slicing(model_for_glyphs, orig_img)
    if len(detections_glyphs) == 0:
        print("No glyph segments detected.")
        sys.exit(0)

    boxes_glyphs = detections_glyphs.xyxy
    confs_glyphs = detections_glyphs.confidence
    classes_glyphs = detections_glyphs.class_id
    polygons_glyphs = masks_to_polygons(detections_glyphs.mask)
else:
    results_glyphs = model_for_glyphs.predict(
        source=image_path,
        verbose=False,
        imgsz=GLYPH_IMGSZ,
        iou=GLYPH_SEGMENTATION_IOU,
        conf=GLYPH_SEGMENTATION_CONF
    )
    res_glyphs = results_glyphs[0]

    if res_glyphs.boxes is None or len(res_glyphs.boxes) == 0:
        print("No glyph segments detected.")
        sys.exit(0)

    boxes_glyphs = to_numpy(res_glyphs.boxes.xyxy)
    confs_glyphs = to_numpy(res_glyphs.boxes.conf)
    classes_glyphs = to_numpy(res_glyphs.boxes.cls)
    polygons_glyphs = None
    if hasattr(res_glyphs, "masks") and res_glyphs.masks is not None:
        polygons_glyphs = res_glyphs.masks.xy

if classes_glyphs is None:
    classes_glyphs = np.zeros(len(boxes_glyphs), dtype=int)

# Sort glyphs vertically top-to-bottom by y1 coordinate
sorted_indices = np.argsort(boxes_glyphs[:, 1])
boxes_glyphs = boxes_glyphs[sorted_indices]
confs_glyphs = confs_glyphs[sorted_indices]
classes_glyphs = classes_glyphs[sorted_indices]
if polygons_glyphs is not None:
    polygons_glyphs = [polygons_glyphs[i] for i in sorted_indices]

# -----------------------------
# Process & Crop Glyphs
# -----------------------------
overlay = orig_img.copy()
alpha = 0.3  # Fill transparency level (0.0 to 1.0)

for i, box in enumerate(tqdm(boxes_glyphs, desc="Processing glyph crops")):
    x1, y1, x2, y2 = map(int, box)
    cls_id = int(classes_glyphs[i])
    color = class_colors[cls_id % len(class_colors)]

    # Clamp coordinates within image boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)

    # Save bounding box crop
    glyph_crop = orig_img[y1:y2, x1:x2]
    if glyph_crop.size > 0:
        cv2.imwrite(str(crop_dir / f"glyph_{i+1:03d}.png"), glyph_crop)

    # Draw filled translucent shape onto overlay
    if polygons_glyphs is not None and len(polygons_glyphs[i]) > 0:
        pts = polygons_glyphs[i].astype(np.int32)
        cv2.fillPoly(overlay, [pts], color=color)
    else:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

# Blend filled overlay with original image
vis_img = cv2.addWeighted(overlay, alpha, orig_img, 1 - alpha, 0)

# -----------------------------
# Draw Crisp Outlines & Labels
# -----------------------------
for i, box in enumerate(boxes_glyphs):
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    cls_id = int(classes_glyphs[i])
    if isinstance(class_names, dict):
        cls_name = class_names.get(cls_id, str(cls_id))
    else:
        cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
    color = class_colors[cls_id % len(class_colors)]

    # Draw solid boundary outline
    if polygons_glyphs is not None and len(polygons_glyphs[i]) > 0:
        pts = polygons_glyphs[i].astype(np.int32)
        cv2.polylines(vis_img, [pts], isClosed=True, color=color, thickness=2)
    else:
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

    # Label glyph class and confidence
    cv2.putText(
        vis_img, f"{cls_name} {confs_glyphs[i]:.2f}", (x1, max(15, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
    )

# -----------------------------
# Save final visual debug image
# -----------------------------
cv2.imwrite(output_path, vis_img)
print(f"Successfully processed {len(boxes_glyphs)} glyphs.")
print(f"Crops saved to: '{crop_dir}/'")
print(f"Visualization saved to: '{output_path}'")
