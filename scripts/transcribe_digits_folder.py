import os
import sys
import subprocess

# -----------------------------
# Usage check
# -----------------------------
if len(sys.argv) < 7:
    print("Usage: python transcribe_folder.py <model_lines> <model_digits> <model_bc> <folder_path> <output_folder> [--clean-masks] [--lines=1,3] [--expand-masks=15] [--save-txt]")
    exit()

model_lines = sys.argv[1]
model_digits = sys.argv[2]
model_bc = sys.argv[3]
folder_path = sys.argv[4]
output_folder = sys.argv[5]
extra_args = sys.argv[6:]  # optional args like --clean-masks etc.

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Collect all image files (you can adjust extensions)
image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".bmp"]
images = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in image_extensions]

for img_name in images:
    input_path = os.path.join(folder_path, img_name)
    output_path = os.path.join(output_folder, img_name)

    cmd = [
        sys.executable,  # python interpreter
        "transcribe.py",
        model_lines,
        model_digits,
        model_bc,
        input_path,
        output_path
    ] + extra_args

    print(f"Processing {img_name} ...")
    subprocess.run(cmd)
