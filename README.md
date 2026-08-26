## Document classification

This branch contains source code for working with models (training and inference) capable of classifying handwritten documents into categories: plaintext, ciphertext, mixed content and cipher key. Additionally, a fully trained models in `.pt` format used for classification within the project are stored here. To properly use these scripts, you need to install the Anaconda environment from the attached YAML file `env.yml`.

### Description of `scripts` folder

- `yolo_predict.py` - This script runs document classification inference (full or quarter image based on the selected model).
- `yolo_train.py` - This script fine-tunes a YOLO11m image classification model on a dataset of document images.
- `yolo_validate.py` - This script performs an end-to-end evaluation of a trained YOLO classification model.

### Description of `models` folder

- `full_document.pt` - trained YOLO8l model for 
- `quarter_document.pt` - trained YOLO8l model for 