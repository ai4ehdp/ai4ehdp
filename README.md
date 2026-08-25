## Line, digit and glyph segmentation

This branch contains source code for working with models (training and inference) capable of segmenting handwritten lines, digits, and glyphs. Additionally, fully trained models in `.pt` format used for inference within the project are stored here.

### `scripts` folder

- `transcribe_digits.py` - Python script for segmenting lines in the input image followed by segmenting digits within the line. Segmented digits are finally transcribed using imitation learning. 
- `transcribe_digits_folder.py` - Python script for segmenting lines in all images in the specified folder followed by segmenting digits within the lines. Segmented digits are finally transcribed using imitation learning.
