import cv2
from ultralytics import YOLO
import sys
import torch
import numpy as np
from shapely.geometry import Polygon
from stable_baselines3.common.policies import ActorCriticPolicy
import os
from tqdm import tqdm   # progress bar
from pathlib import Path

# -----------------------------
# Check command-line arguments
# -----------------------------
if len(sys.argv) < 6:
    print("Usage:\n transcribe_by_line_segmentation.py \n\t<model_lines> \n\t<model_digits> \n\t<model_bc> \n\t<image_path> \n\t<output_img_path> \n\t[--clean-masks] \n\t[--expand-masks=15] \n\t[--lines=1,3,5] \n\t[--save-txt]")
    exit()

# -----------------------------
# Utility functions
# -----------------------------
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x

def bbox_relative_center(digit_bbox, line_bbox):
    x1, y1, x2, y2 = line_bbox
    w = x2 - x1
    h = y2 - y1
    if w == 0 or h == 0:
        return (0.5, 0.5)
    minx, miny, maxx, maxy = digit_bbox
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    rel_x = (center_x - x1) / w
    rel_y = (center_y - y1) / h
    return (rel_x, rel_y)

# -----------------------------
# Imitation learning helpers
# -----------------------------
def predict_action(policy, obs):
    action, _ = policy.predict(obs, deterministic=True)
    return int(action)

def predict_reading_order(centers, max_candidates, policy):
    N = len(centers)
    visited = [0]
    current = 0
    while current < N - 1:
        candidates = centers[current + 1 : current + 1 + max_candidates]
        if len(candidates) == 0:
            break
        obs = np.zeros((max_candidates, 2), dtype=np.float32)
        for i, c in enumerate(candidates):
            obs[i] = np.array(c, dtype=np.float32)
        action = predict_action(policy, obs)
        if action >= len(candidates):
            action = len(candidates) - 1
        next_idx = current + 1 + action
        if next_idx in visited:
            break
        visited.append(next_idx)
        current = next_idx
    return visited
        
    
def expand_mask(mask, base_dilate=5, extra=10):
    """
    Expand convex regions of a binary mask more than rest.

    mask: uint8 mask, 0/255
    base_dilate: general dilation for whole mask
    extra_convex: extra dilation for convex hull region
    """
    # 1. Basic dilation for the whole mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base_dilate, base_dilate))
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)
    
    return mask_dilated

    # 2. Compute convex hull of mask
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if not contours:
        # return mask_dilated
    # hull = cv2.convexHull(np.vstack(contours))
    # hull_mask = np.zeros_like(mask)
    # cv2.fillPoly(hull_mask, [hull], 255)

    #3. Extra dilation for convex hull area
    # convex_diff = cv2.subtract(hull_mask, mask)  # convex-only areas
    # kernel_convex = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (extra, extra))
    # convex_diff_dilated = cv2.dilate(convex_diff, kernel_convex, iterations=1)

    #4. Merge with base dilated mask
    # mask_final = cv2.bitwise_or(mask_dilated, convex_diff_dilated)
    # return mask_final
    

USE_CLEAN_MASKS = "--clean-masks" in sys.argv
USE_EXPAND_MASKS = False
SAVE_TXT = "--save-txt" in sys.argv

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
CLASS_NAMES = ['5', '8', '3', '1', '4', '2', '0', '7', '6', '9']
MAX_CAND = 20
COLORS = [
    (0,255,0), (0,255,255), (255,0,0), (255,0,255), (0,128,255),
    (0,0,255), (128,0,128), (255,255,0), (128,128,0), (255,128,0)
]

# -----------------------------
# Load behavioral cloning policy
# -----------------------------
policy = ActorCriticPolicy.load(sys.argv[3])
policy.eval()

# -----------------------------
# Load YOLO models
# -----------------------------
model_for_lines = YOLO(sys.argv[1])
model_for_digits = YOLO(sys.argv[2])
image_path = sys.argv[4]
output_path = sys.argv[5]

# -----------------------------
# Parse selected lines early (1-based indices -> convert to 0-based)
# We'll finalize them after we know how many detected lines exist.
# -----------------------------
lines_to_draw_raw = None
for arg in sys.argv:
    if arg.startswith("--lines"):
        try:
            nums = arg.split("=")[1]
            lines_to_draw_raw = [int(x.strip()) - 1 for x in nums.split(",")]
        except Exception:
            lines_to_draw_raw = None

# -----------------------------
# Ensure output folders exist
# -----------------------------
os.makedirs('line_crops', exist_ok=True)

# -----------------------------
# Read image
# -----------------------------
orig_img = cv2.imread(image_path)
if orig_img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")
img_h, img_w = orig_img.shape[:2]
vis_img = orig_img.copy()

# -----------------------------
# Segment lines in the image
# -----------------------------
results_lines = model_for_lines.predict(source=image_path, verbose=False, imgsz=LINE_IMGSZ, iou=LINE_SEGMENTATION_IOU, conf=LINE_SEGMENTATION_CONF)
res_lines = results_lines[0]

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

# -----------------------------
# Finalize effective lines_to_draw (None means "all lines")
# -----------------------------
total_lines = len(boxes_lines)
if lines_to_draw_raw is None:
    lines_to_draw = None
else:
    # filter indices that are in range
    valid = [x for x in lines_to_draw_raw if 0 <= x < total_lines]
    if len(valid) == 0:
        # fallback to all lines (same behaviour as before when invalid)
        lines_to_draw = None
    else:
        lines_to_draw = valid

transcriptions = []

# Prepare storage for cleaned full-size masks / polygons (so we can draw them on the right image later)
cleaned_masks_full_list = [None] * len(boxes_lines)
cleaned_polygons_list = [None] * len(boxes_lines)

# -----------------------------
# Process each detected line WITH PROGRESS BAR
# -----------------------------
for i, (line_box, line_conf) in tqdm(
        enumerate(zip(boxes_lines, confs_lines), start=0),
        total=len(boxes_lines),
        desc="Processing lines"):

    x1, y1, x2, y2 = line_box.astype(int)
    min_x, max_x = x1, x2
    min_y, max_y = y1, y2

    # --------------------------------------------------------------------------
    # APPLY MORPHOLOGICAL CLEANING HERE (NEW LOCATION)
    # We always construct a full-size mask from the polygon (if polygon exists)
    # and then optionally apply cleaning. We store the resulting full-size mask
    # and polygon into cleaned_* lists for later use on the right-side image.
    # --------------------------------------------------------------------------
    if polygons_lines and len(polygons_lines) > i:
        poly_abs = np.array(polygons_lines[i], dtype=np.int32)

        # create FULL SIZE mask first (unprocessed)
        line_mask_full = np.zeros(orig_img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(line_mask_full, [poly_abs], 255)

        # Save the unmodified polygon/mask in case cleaning is disabled or we want fall-back
        cleaned_masks_full_list[i] = line_mask_full.copy()
        cleaned_polygons_list[i] = poly_abs.copy()

        # Optionally perform morphological cleaning (controlled by flag)
        if USE_CLEAN_MASKS:
            kernel_size = 15
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            line_mask_full = cv2.morphologyEx(line_mask_full, cv2.MORPH_OPEN, kernel)

            # keep only largest connected component
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(line_mask_full, connectivity=8)
            if num_labels > 1:
                largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                line_mask_full = np.where(labels == largest, 255, 0).astype(np.uint8)

            # recompute polygon from cleaned mask
            contours, _ = cv2.findContours(line_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                poly_abs = contours[0].reshape(-1,2)

            # store cleaned results (overwrite the previous saved unmodified mask/polygon)
            cleaned_masks_full_list[i] = line_mask_full.copy()
            cleaned_polygons_list[i] = poly_abs.copy()
        
        # -----------------------------
        # OPTIONAL MASK EXPANSION
        # -----------------------------
        if USE_EXPAND_MASKS:
            line_mask_full = expand_mask(line_mask_full, base_dilate=dilate_size, extra=dilate_size//4)

            # recompute polygon from expanded mask
            contours, _ = cv2.findContours(line_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                poly_abs = contours[0].reshape(-1,2)

            # store updated mask/polygon
            cleaned_masks_full_list[i] = line_mask_full.copy()
            cleaned_polygons_list[i] = poly_abs.copy()

        # --------------------------------------------------------------------------
        # crop bounds from the (potentially cleaned) polygon
        # --------------------------------------------------------------------------
        min_x = max(np.min(poly_abs[:, 0]), 0)
        max_x = min(np.max(poly_abs[:, 0]), orig_img.shape[1]-1)
        min_y = max(np.min(poly_abs[:, 1]), 0)
        max_y = min(np.max(poly_abs[:, 1]), orig_img.shape[0]-1)

        line_crop = orig_img[min_y:max_y, min_x:max_x].copy()

        # convert polygon to crop coordinates
        poly_crop = poly_abs.copy()
        poly_crop[:, 0] -= min_x
        poly_crop[:, 1] -= min_y

        mask = np.zeros(line_crop.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [poly_crop], 255)

        # compute border color
        border_colors = []
        for px, py in poly_crop:
            if 0 <= px < line_crop.shape[1] and 0 <= py < line_crop.shape[0]:
                border_colors.append(line_crop[py, px])
        border_color = np.mean(border_colors, axis=0).astype(np.uint8) if border_colors else [0,0,0]

        line_crop[mask == 0] = border_color

        cv2.imwrite("line_crops/crop"+str(i)+".jpg", line_crop)

    else:
        # no polygon available, fallback to bbox crop
        line_crop = orig_img[y1:y2, x1:x2].copy()

    # -----------------------------
    # LEFT SIDE DRAWING filtering (Option A: show only selected lines)
    # -----------------------------
    draw_this_line = (lines_to_draw is None) or (i in lines_to_draw)

    # Draw line bounding box on left only if selected
    if draw_this_line:
        cv2.rectangle(vis_img, (min_x, min_y), (max_x, max_y), (255,255,255), 2)

        # label
        line_label = f"Line {i+1}"
        (font_w, font_h), _ = cv2.getTextSize(line_label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        label_x = max_x - font_w - 5
        label_y = max(min_y + font_h + 5, font_h + 5)
        cv2.rectangle(vis_img, (label_x, label_y - font_h - 4),
                      (label_x + font_w, label_y), (0, 0, 0), -1)
        cv2.putText(vis_img, line_label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)

    # -----------------------------
    # Detect digits in cropped line
    # -----------------------------
    results_digits = model_for_digits.predict(source=line_crop, verbose=False, iou=DIGIT_SEGMENTATION_IOU, conf=DIGIT_SEGMENTATION_CONF, agnostic_nms=True)
    res_digits = results_digits[0]

    line_digits_for_transcription = []

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

            # Draw digits on left only if this line is selected
            if draw_this_line:
                cv2.rectangle(overlay, (ox1, oy1), (ox2, oy2), color, 2)

                if polygons_digits and len(polygons_digits) > idx:
                    poly = np.array(polygons_digits[idx], dtype=np.int32)
                    cv2.fillPoly(overlay, [poly + np.array([min_x, min_y])], color)

                label = f'{cls_name} {d_conf:.2f}'
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis_img, (ox1, oy1 - h - 4), (ox1 + w, oy1), color, -1)
                cv2.putText(vis_img, label, (ox1, oy1 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            if polygons_digits and len(polygons_digits) > idx:
                poly = np.array(polygons_digits[idx], dtype=np.float32)
                try:
                    shapely_poly = Polygon(poly)
                    rep_point = shapely_poly.representative_point()
                    rep_x = rep_point.x
                except Exception:
                    rep_x = (dx1 + dx2) / 2
            else:
                rep_x = (dx1 + dx2) / 2

            line_digits_for_transcription.append(
                (rep_x, cls_name, bbox_relative_center([ox1, oy1, ox2, oy2], [min_x, min_y, max_x, max_y]))
            )

        # apply overlay only if we drew anything for this line
        if draw_this_line:
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0, vis_img)

    # -----------------------------
    # Sort digits left to right based on representative point
    # -----------------------------
    line_digits_for_transcription.sort(key=lambda x: x[0])
    relative_centers = [t[2] for t in line_digits_for_transcription]
    pred_order = predict_reading_order(relative_centers, MAX_CAND, policy)

    # transcription
    line_text_digits = []
    if len(line_digits_for_transcription) > 0:
        for ii in range(len(pred_order)):
            line_text_digits.append(line_digits_for_transcription[pred_order[ii]][1])
    line_text = "".join(line_text_digits)
    transcriptions.append(line_text)

    # Draw transcription text on left only if this line is selected
    if line_text and draw_this_line:
        font_scale = 1.2
        thickness = 2
        (w, h), _ = cv2.getTextSize(line_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = min_x
        text_y = max(min_y - 5, h + 2)
        cv2.rectangle(vis_img, (text_x, text_y - h - 2), (text_x + w, text_y), (0,0,0), -1)
        cv2.putText(vis_img, line_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness, cv2.LINE_AA)

# -----------------------------
# Left-side image: selected line masks (use cleaned masks if available)
# -----------------------------
left_vis = orig_img.copy()

# lines_to_draw has already been computed earlier (None => all lines)
if polygons_lines:
    print("\nDetected lines:")
    for i, poly in enumerate(polygons_lines):
        print(f"{i+1}: Line {i+1}, confidence {confs_lines[i]:.2f}")

    for i in (range(len(polygons_lines)) if lines_to_draw is None else lines_to_draw):
        # Prefer cleaned full-size mask if available (we created one during processing)
        if cleaned_masks_full_list[i] is not None:
            line_mask = cleaned_masks_full_list[i].copy()
            poly_np = cleaned_polygons_list[i].copy() if cleaned_polygons_list[i] is not None else np.array(polygons_lines[i], dtype=np.int32)
        else:
            # fallback to original polygon mask
            poly_np = np.array(polygons_lines[i], dtype=np.int32)
            line_mask = np.zeros((orig_img.shape[0], orig_img.shape[1]), dtype=np.uint8)
            cv2.fillPoly(line_mask, [poly_np], 255)

        # Overlay cleaned mask (semi-transparent)
        overlay = left_vis.copy()
        overlay[line_mask > 0] = (255, 255, 0)  # yellow-ish overlay
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, left_vis, 1 - alpha, 0, left_vis)

        # Draw contour of the mask (cleaned contour if cleaned mask used)
        contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(left_vis, contours, -1, (255, 255, 0), 3)

        line_conf = confs_lines[i]
        # compute label position from polygon
        x_text, y_text = np.min(poly_np[:,0]), np.min(poly_np[:,1]) - 5
        cv2.putText(left_vis, f"{line_conf:.2f}", (x_text, max(y_text, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)


# -----------------------------
# Concatenate left and right
# -----------------------------
combined_vis = np.concatenate([left_vis, vis_img], axis=1)
path_obj =  Path(output_path)
output_path_left = path_obj.with_name(f"{path_obj.stem}_lines{path_obj.suffix}")
output_path_right = path_obj.with_name(f"{path_obj.stem}_digits{path_obj.suffix}")

cv2.imwrite(str(output_path_left), left_vis)
cv2.imwrite(str(output_path_right), vis_img)
cv2.imwrite(output_path, combined_vis)
print(f"\nSaved visualization to {output_path}")

print("\nTranscription:\n")
for tl in transcriptions:
        print(tl)
        
# -----------------------------
# Save transcription to txt file if requested
# -----------------------------
if SAVE_TXT:
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    txt_path = os.path.join(os.path.dirname(output_path), base_name + ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for line in transcriptions:
            f.write(line + "\n")
    print(f"Saved transcription to {txt_path}")
