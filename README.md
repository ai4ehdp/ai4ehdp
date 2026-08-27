## Line, digit and glyph segmentation

This branch contains source code for working with models (training and inference) capable of segmenting handwritten lines, digits, and glyphs. Additionally, fully trained models in `.pt` format used for inference within the project are stored here. To properly use these scripts, you need to install the Anaconda environment from the attached YAML file `env.yml`.

### Description of `scripts` folder

- `digits/segment_digits.py` - Python script for segmenting lines in the input image followed by segmenting digits within the line.
- `digits/segment_digits_folder.py` - Python script for segmenting lines in all images in the specified folder followed by segmenting digits within the lines.
- `digits/train_cli_command.txt` - CLI command for training YOLO model for segmenting digits in lines.
- `glyphs/segment_glyphs.py` - Python script for segmenting glyphs in the input image.
- `glyphs/segment_glyphs_folder.py` - Python script for segmenting glyphs in all images in the specified folder.
- `glyphs/train_cli_command.txt` - CLI command for training YOLO model for segmenting glyphs in images.
- `lines/segment_lines.py` - Python script for segmenting lines in the input image. 
- `lines/segment_lines_folder.py` - Python script for segmenting lines in all images in the specified folder.

### Description of `models` folder

- `digits/segment_digits_in_lines.pt` - trained YOLO8l model for segmenting digits in lines
- `glyphs/segment_glyphs.pt` - trained YOLO11l model for segmenting glyphs in the input document
- `lines/segment_lines.pt` - trained YOLO8l model for segmenting lines in the input document
