import cv2
from ultralytics import YOLO
import sys
import torch
import numpy as np
from tqdm import tqdm   # progress bar
from pathlib import Path
import os

# -----------------------------
# Check command-line arguments
# -----------------------------
if len(sys.argv) < 5:
    print("Usage:\n segment_digits.py \n\t<model_lines> \n\t<model_digits> \n\t<input_folder_path> \n\t<output_folder_path> \n\t[--clean-masks] \n\t[--expand-masks=15] \n\t[--lines=1,3,5]")
    exit()

# -----------------------------
# Utility functions
# -----------------------------
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x

def expand_mask(mask, base_dilate=5):
    """
    Expand regions of a binary mask.
    mask: uint8 mask, 0/255
    base_dilate: general dilation for whole mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base_dilate, base_dilate))
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)
    return mask_dilated

USE_CLEAN_MASKS = "--clean-masks" in sys.argv
USE_EXPAND_MASKS = False

LINE_SEGMENTATION_IOU = 0.50
LINE_SEGMENTATION_CONF = 0.50

DIGIT_SEGMENTATION_IOU = 0.40
DIGIT_SEGMENTATION_CONF = 0.20

LINE_IMGSZ = 1024

dilate_size = 15  # default

for arg in sys.argv:
    if arg.startswith("--expand-masks"):
        USE_EXPAND_MASKS = True
        if "=" in arg:
            try:
                dilate_size = int(arg.split("=")[1])
            except ValueError:
                print("Invalid --expand-masks value, using default 15")

# -----------------------------
# Class names and visualization
# -----------------------------
CLASS_NAMES = ['0','1','2','3','4','5','6','7','8','9']
COLORS = [
    (0,255,0), (0,255,255), (255,0,0), (255,0,255), (0,128,255),
    (0,0,255), (128,0,128), (255,255,0), (128,128,0), (255,128,0)
]

# -----------------------------
# Load YOLO models
# -----------------------------
model_for_lines = YOLO(sys.argv[1])
model_for_digits = YOLO(sys.argv[2])
input_folder = Path(sys.argv[3])
output_folder = Path(sys.argv[4])

# Ensure output directory exists
output_folder.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Parse selected lines early (1-based indices -> convert to 0-based)
# -----------------------------
lines_to_draw_raw = None
for arg in sys.argv:
    if arg.startswith("--lines"):
        try:
            nums = arg.split("=")[1]
            lines_to_draw_raw = [int(x.strip()) - 1 for x in nums.split(",")]
        except Exception:
            lines_to_draw_raw = None

# Supported image extensions
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
image_paths = [p for p in input_folder.iterdir() if p.suffix.lower() in VALID_EXTENSIONS]

if not image_paths:
    print(f"No valid images found in: {input_folder}")
    exit()

print(f"Found {len(image_paths)} image(s) to process.")

# -----------------------------
# Process each image in folder
# -----------------------------
for image_path in tqdm(image_paths, desc="Processing folder"):
    orig_img = cv2.imread(str(image_path))
    if orig_img is None:
        print(f"Warning: Could not read image {image_path}, skipping.")
        continue

    img_h, img_w = orig_img.shape[:2]
    vis_img = orig_img.copy()

    # Segment lines in the image
    results_lines = model_for_lines.predict(source=str(image_path), verbose=False, imgsz=LINE_IMGSZ, iou=LINE_SEGMENTATION_IOU, conf=LINE_SEGMENTATION_CONF)
    res_lines = results_lines[0]

    if len(res_lines.boxes) == 0:
        continue

    boxes_lines = to_numpy(res_lines.boxes.xyxy)
    confs_lines = to_numpy(res_lines.boxes.conf)

    # Sort lines vertically
    sorted_indices = np.argsort(boxes_lines[:, 1])
    boxes_lines = boxes_lines[sorted_indices]
    confs_lines = confs_lines[sorted_indices]

    masks_lines = getattr(res_lines, "masks", None)
    polygons_lines = None
    if masks_lines is not None and hasattr(masks_lines, "xy"):
        polygons_lines = masks_lines.xy
        polygons_lines = [polygons_lines[i] for i in sorted_indices]

    # Finalize effective lines_to_draw (None means "all lines")
    total_lines = len(boxes_lines)
    if lines_to_draw_raw is None:
        lines_to_draw = None
    else:
        valid = [x for x in lines_to_draw_raw if 0 <= x < total_lines]
        lines_to_draw = valid if len(valid) > 0 else None

    cleaned_masks_full_list = [None] * len(boxes_lines)
    cleaned_polygons_list = [None] * len(boxes_lines)

    # Process each line
    for i, (line_box, line_conf) in enumerate(zip(boxes_lines, confs_lines)):
        x1, y1, x2, y2 = line_box.astype(int)
        min_x, max_x = x1, x2
        min_y, max_y = y1, y2

        if polygons_lines and len(polygons_lines) > i:
            poly_abs = np.array(polygons_lines[i], dtype=np.int32)

            line_mask_full = np.zeros(orig_img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(line_mask_full, [poly_abs], 255)

            cleaned_masks_full_list[i] = line_mask_full.copy()
            cleaned_polygons_list[i] = poly_abs.copy()

            if USE_CLEAN_MASKS:
                kernel_size = 15
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                line_mask_full = cv2.morphologyEx(line_mask_full, cv2.MORPH_OPEN, kernel)

                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(line_mask_full, connectivity=8)
                if num_labels > 1:
                    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    line_mask_full = np.where(labels == largest, 255, 0).astype(np.uint8)

                contours, _ = cv2.findContours(line_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    poly_abs = contours[0].reshape(-1, 2)

                cleaned_masks_full_list[i] = line_mask_full.copy()
                cleaned_polygons_list[i] = poly_abs.copy()
            
            if USE_EXPAND_MASKS:
                line_mask_full = expand_mask(line_mask_full, base_dilate=dilate_size)

                contours, _ = cv2.findContours(line_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    poly_abs = contours[0].reshape(-1, 2)

                cleaned_masks_full_list[i] = line_mask_full.copy()
                cleaned_polygons_list[i] = poly_abs.copy()

            min_x = max(np.min(poly_abs[:, 0]), 0)
            max_x = min(np.max(poly_abs[:, 0]), orig_img.shape[1] - 1)
            min_y = max(np.min(poly_abs[:, 1]), 0)
            max_y = min(np.max(poly_abs[:, 1]), orig_img.shape[0] - 1)

            line_crop = orig_img[min_y:max_y, min_x:max_x].copy()

            poly_crop = poly_abs.copy()
            poly_crop[:, 0] -= min_x
            poly_crop[:, 1] -= min_y

            mask = np.zeros(line_crop.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [poly_crop], 255)

            border_colors = []
            for px, py in poly_crop:
                if 0 <= px < line_crop.shape[1] and 0 <= py < line_crop.shape[0]:
                    border_colors.append(line_crop[py, px])
            border_color = np.mean(border_colors, axis=0).astype(np.uint8) if border_colors else [0, 0, 0]

            line_crop[mask == 0] = border_color
        else:
            line_crop = orig_img[y1:y2, x1:x2].copy()

        draw_this_line = (lines_to_draw is None) or (i in lines_to_draw)

        if draw_this_line:
            cv2.rectangle(vis_img, (min_x, min_y), (max_x, max_y), (255, 255, 255), 2)

            line_label = f"Line {i+1}"
            (font_w, font_h), _ = cv2.getTextSize(line_label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
            label_x = max_x - font_w - 5
            label_y = max(min_y + font_h + 5, font_h + 5)
            cv2.rectangle(vis_img, (label_x, label_y - font_h - 4),
                          (label_x + font_w, label_y), (0, 0, 0), -1)
            cv2.putText(vis_img, line_label, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        # Detect digits in cropped line
        results_digits = model_for_digits.predict(source=line_crop, verbose=False, iou=DIGIT_SEGMENTATION_IOU, conf=DIGIT_SEGMENTATION_CONF, agnostic_nms=True)
        res_digits = results_digits[0]

        if len(res_digits.boxes) > 0:
            boxes_digits = to_numpy(res_digits.boxes.xyxy)
            confs_digits = to_numpy(res_digits.boxes.conf)
            classes_digits = to_numpy(res_digits.boxes.cls)

            masks_digits = getattr(res_digits, "masks", None)
            polygons_digits = masks_digits.xy if (masks_digits is not None and hasattr(masks_digits, "xy")) else None

            overlay = vis_img.copy()

            for idx, (d_box, d_conf, d_cls) in enumerate(zip(boxes_digits, confs_digits, classes_digits)):
                dx1, dy1, dx2, dy2 = d_box.astype(int)
                cls_id = int(d_cls)
                cls_name = CLASS_NAMES[cls_id]
                color = COLORS[cls_id % len(COLORS)]

                ox1 = min_x + dx1
                oy1 = min_y + dy1
                ox2 = min_x + dx2
                oy2 = min_y + dy2

                if draw_this_line:
                    cv2.rectangle(overlay, (ox1, oy1), (ox2, oy2), color, 2)

                    if polygons_digits and len(polygons_digits) > idx:
                        poly = np.array(polygons_digits[idx], dtype=np.int32)
                        cv2.fillPoly(overlay, [poly + np.array([min_x, min_y])], color)

                    label = f'{cls_name} {d_conf:.2f}'
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(vis_img, (ox1, oy1 - h - 4), (ox1 + w, oy1), color, -1)
                    cv2.putText(vis_img, label, (ox1, oy1 - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            if draw_this_line:
                alpha = 0.4
                cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0, vis_img)

    # Left-side image: selected line masks
    left_vis = orig_img.copy()

    if polygons_lines:
        for i in (range(len(polygons_lines)) if lines_to_draw is None else lines_to_draw):
            if cleaned_masks_full_list[i] is not None:
                line_mask = cleaned_masks_full_list[i].copy()
                poly_np = cleaned_polygons_list[i].copy() if cleaned_polygons_list[i] is not None else np.array(polygons_lines[i], dtype=np.int32)
            else:
                poly_np = np.array(polygons_lines[i], dtype=np.int32)
                line_mask = np.zeros((orig_img.shape[0], orig_img.shape[1]), dtype=np.uint8)
                cv2.fillPoly(line_mask, [poly_np], 255)

            overlay = left_vis.copy()
            overlay[line_mask > 0] = (255, 255, 0)  # yellow overlay
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, left_vis, 1 - alpha, 0, left_vis)

            contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(left_vis, contours, -1, (255, 255, 0), 3)

            line_conf = confs_lines[i]
            x_text, y_text = np.min(poly_np[:, 0]), np.min(poly_np[:, 1]) - 5
            cv2.putText(left_vis, f"{line_conf:.2f}", (x_text, max(y_text, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    # Save visual outputs into output folder using stem of original filename
    combined_vis = np.concatenate([left_vis, vis_img], axis=1)
    stem = image_path.stem
    ext = image_path.suffix

    output_left = output_folder / f"{stem}_lines{ext}"
    output_right = output_folder / f"{stem}_digits{ext}"
    output_combined = output_folder / f"{stem}_combined{ext}"

    cv2.imwrite(str(output_left), left_vis)
    cv2.imwrite(str(output_right), vis_img)
    cv2.imwrite(str(output_combined), combined_vis)

print(f"\nProcessing complete! All visualizations saved to: {output_folder}")