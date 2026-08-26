## Digit transcription

This branch contains source code for working with models (training and inference) capable of transcribing handwritten digits in historical encrypted documents using imitation learning. Additionally, fully trained models in `.zip` and `.pt` format used for transcription within the project are stored here. To properly use these scripts, you need to install the Anaconda environment from the attached YAML file `env.yml`.

### Description of `scripts` folder

- `transcribe_digits.py` - Python script for segmenting lines in the input image followed by segmenting digits within the line. Finally, the digits are transcribed using imitation learning.
- `transcribe_digits_folder.py` - Python script for segmenting lines in all images in the specified folder followed by segmenting digits within the lines. Finally, the digits in all images are transcribed using imitation learning.

### Description of `models` folder

- `segment_digits_in_lines.pt` - trained YOLO8l model for segmenting digits in lines
- `segment_lines.pt` - trained YOLO8l model for segmenting lines in the input document
- `bc_policy.zip` - trained imitation learning model (behavioral cloning policy) responsible for digit transcription