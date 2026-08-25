## Line, digit and glyph segmentation

This branch contains source code for working with models (training and inference) capable of segmenting handwritten lines, digits, and glyphs. Additionally, fully trained models in `.pt` format used for inference within the project are stored here. To properly use these scripts, you need to install the Anaconda environment from the attached YAML file `env.yml`.

### Description of `scripts` folder

- `digits/transcribe_digits.py` - Python script for segmenting lines in the input image followed by segmenting digits within the line. Segmented digits are finally transcribed using imitation learning. 
- `digits/transcribe_digits_folder.py` - Python script for segmenting lines in all images in the specified folder followed by segmenting digits within the lines. Segmented digits are finally transcribed using imitation learning.
- `digits/train_cli_command.txt` - CLI command for training YOLO model for segmenting digits in lines

### Description of `models` folder

- `lines/segment_lines.pt` - trained YOLO8l model for segmenting lines in the input document
- `digits/bc_policy.zip` - trained imitation learning model based on behavioral cloning for transcription of digits
