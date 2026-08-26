from ultralytics import YOLO
from pathlib import Path
import json
import  csv
import os

# Configuration

model = YOLO("yolo26m-seg.pt")


data_yaml = "hhcs_cipher_key_layouts_seg/hhcs_cipher_key_layouts_seg_smote/data.yaml"
img_size = 640
epochs = 100
batch = 32
project_name = "HCKC"
run = "yolo26_M_ourA"

# Training

train_results = model.train(
    data=data_yaml,
    epochs=100,
    imgsz=img_size,
    batch=32,
    project=project_name,
    name=run,
    pretrained=True,



    mosaic = 0.0,
    mixup = 0.0,
    flipud = 0.1,
    fliplr = 0.1,
    degrees = 15,
    shear = 5,
    scale = 0.25,
    translate = 0.0,
    close_mosaic = 0,
    copy_paste=0.0,
    erasing = 0.0,
    cutmix = 0.0,
    hsv_h = 0.015,
    hsv_s = 0.7,
    hsv_v = 0.4,

)

save_dir = Path(train_results.save_dir)


best_pt = save_dir / "weights" / "best.pt"
last_pt = save_dir / "weights" / "last.pt"
weights = best_pt if best_pt.exists() else last_pt

# Evaluation

eval_model = YOLO(str(weights))

test_results = eval_model.val(
    data=data_yaml,
    split="test",
    imgsz=img_size,
    batch=batch,
    verbose=True
)



rd = getattr(test_results, "results_dict", {}) or {}
overall = {
    "weights_used": str(weights),
    "box/map50": float(rd.get("metrics/mAP50(B)", 0.0)),
    "box/map50-95": float(rd.get("metrics/mAP50-95(B)", 0.0)),
    "seg/map50": float(rd.get("metrics/mAP50(M)", 0.0)),
    "seg/map50-95": float(rd.get("metrics/mAP50-95(M)", 0.0)),
}

# Per-class metrics

names = getattr(test_results, "names", None) or getattr(eval_model, "names", None) or {}


def _to_list(x):
    if x is None:
        return None
    try:
        return [float(v) for v in x]
    except Exception:
        return None

def extract_per_class(metric_obj):

    if metric_obj is None:
        return {}

    ap50 = _to_list(getattr(metric_obj, "ap50", None))
    ap = _to_list(getattr(metric_obj, "ap", None))

    out = {}
    if ap50 is None and ap is None:
        return out

    n = 0
    if ap50 is not None:
        n = max(n, len(ap50))
    if ap is not None:
        n = max(n, len(ap))

    for i in range(n):
        cname = names.get(i, str(i))
        out[cname] = {}
        if ap50 is not None and i < len(ap50):
            out[cname]["map50"] = ap50[i]
        if ap is not None and i < len(ap):
            out[cname]["map50-95"] = ap[i]
    return out

per_class = {
    "weights_used": str(weights),
    "box": extract_per_class(getattr(test_results, "box", None)),
    "seg": extract_per_class(getattr(test_results, "seg", None)),
}


# Saving

save_dir.mkdir(parents=True, exist_ok=True)


with open(save_dir / "test_metrics_overall.json", "w", encoding="utf-8") as f:
    json.dump(overall, f, ensure_ascii=False, indent=2)

with open(save_dir / "test_metrics_per_class.json", "w", encoding="utf-8") as f:
    json.dump(per_class, f, ensure_ascii=False, indent=2)

csv_path = save_dir / "test_metrics_per_class.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["task", "class", "map50", "map50-95"])

    for task in ("box", "seg"):
        for cls_name, vals in per_class.get(task, {}).items():
            w.writerow([
                task,
                cls_name,
                vals.get("map50", ""),
                vals.get("map50-95", "")
            ])

print("\nUložené do runu:")
print(f"- {save_dir / 'test_metrics_overall.json'}")
print(f"- {save_dir / 'test_metrics_per_class.json'}")
print(f"- {csv_path}")
print(" Overall:", overall)