# HCKC – YOLO26M-Seg Training & Evaluation

Script for training and evaluating a YOLO26-M segmentation model on the
cipher key layouts dataset (SMOTE-augmented). Here you can find best-performing configuration and model weights from:
**HCKC: a Dataset of Historical Cipher Key Component Annotations for Machine Learning Tasks** publication

## Requirements

- Python 3.9+
- NVIDIA GPU with CUDA (recommended; CPU training will be extremely slow)
- Required packages:

```bash
pip install ultralytics
```

`ultralytics` automatically installs `torch`, `torchvision`, and other
dependencies. For GPU support, make sure you have a PyTorch version
matching your CUDA version installed (see https://pytorch.org/get-started/locally/).

Optional, if you run into CUDA issues:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Dataset structure

The script expects a dataset in YOLO segmentation format, defined via
`data.yaml`:

```
hhcs_cipher_key_layouts_seg/
└── hhcs_cipher_key_layouts_seg_smote/
    ├── data.yaml
    ├── train/
    ├── val/
    └── test/
```

`data.yaml` must contain paths to the `train`, `val`, and `test` splits and
the list of classes (`names`).

## Model weights

The script uses pretrained weights `yolo26m-seg.pt`. Ultralytics will
automatically download them on first run if not already available locally.
For offline environments, download them beforehand and place them in the
working directory.

## Running the script

```bash
python train_eval.py
```

(replace with your script's actual filename if different)

The script performs:

1. **Training** of the YOLO26-M-seg model for 100 epochs with the defined
   augmentations (mosaic disabled, flip, rotation, shear, scale, HSV
   adjustments).
2. **Best-weights selection** – uses `best.pt` from the `weights/` folder,
   falling back to `last.pt` if `best.pt` doesn't exist.
3. **Evaluation on the test split** using `model.val(split="test")`.
4. **Metrics export**:
   - `test_metrics_overall.json` – overall mAP50 and mAP50-95 for box and seg
   - `test_metrics_per_class.json` – per-class metrics
   - `test_metrics_per_class.csv` – per-class metrics in CSV format

All outputs are saved to:

```
<project_name>/<run>/
```

## Console output

After completion, the script prints the paths to the saved files and the
overall metrics directly to the terminal.
