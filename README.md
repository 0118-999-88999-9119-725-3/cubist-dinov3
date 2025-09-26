# DINOv3 with CubistMerge

This repository is a fork of the original DINOv3 implementation with integrated token merging capabilities.

For the original DINOv3 documentation, please see [DINOV3_README.md](DINOV3_README.md).

## Environment Setup

### Prerequisites
Follow the environment setup and model download instructions in [DINOV3_README.md](DINOV3_README.md).

### Installation
After setting up the base environment, install this repository to pick up the CubistMerge modifications:

```bash
pip3 install .
```

## Setup

Before running the scripts, you need to configure the paths in both test scripts:

**Common variables:**
- `BACKBONE_WEIGHTS`: Path to the DINOv3 backbone pretrained weights file

**For `test_classifier.py`**:
- `WEIGHTS`: Path to the DINOv3 classification head weights file
- `IMAGENET_VAL_DIR`: Path to ImageNet validation dataset directory

**For `test_detection.py`**:
- `WEIGHTS`: Path to the DINOv3 detection head weights file
- `RESULT_PATH`: Path where detection results JSON file will be saved
- `COCO_ANN_FILE`, `COCO_IMG_DIR`: Path to COCO annotation file and image directory

## Usage

Our implementation supports three token merging methods: `cume`, `tome`, and `expedite`.

The `tome` implementation is adapted from https://github.com/facebookresearch/ToMe and `expedite` is adapted from https://github.com/Expedit-LargeScale-Vision-Transformer.

### Classification

To run the baseline model without any token merging:
```bash
python3 test_classifier.py -s <iter>
```

The `-s <iter>` parameter allows you to early stop evaluation at a specific iteration (optional, without `-s` runs full evaluation).

To run with token merging:
```bash
python3 test_classifier.py -m <method> -l <layer> -r <r>
```

**Token merging parameters:**
- `-m <method>`: Token merging method (`cume`, `tome`, or `expedite`)
- `-l <layer>`: Layer to apply token merging
- `-r <r>`: Token reduction rate (integer values: 1, 2, 3, etc.)

Given H × W input tokens, this reduces to (H - r) × (W - r) tokens.

**Example:**
```bash
python3 test_classifier.py -s 100  # baseline
python3 test_classifier.py -s 100 -m cume -l 20 -r 1
python3 test_classifier.py -s 100 -m tome -l 20 -r 1
python3 test_classifier.py -s 100 -m expedite -l 20 -r 1
```

### Detection

For detection tasks, install pycocotools

```bash
pip3 install pycocotools
```

To run the baseline model without any token merging:
```bash
python3 test_detection.py -s <iter>
```

The `-s <iter>` parameter allows you to early stop evaluation at a specific iteration (optional, without `-s` runs full evaluation).

To run with token merging:
```bash
python3 test_detection.py -m <method> -l <layer> -r <r>
```

**Token merging parameters:**
- `-m <method>`: Token merging method (`cume`, `tome`, or `expedite`)
- `-l <layer>`: Layer to apply token merging
- `-r <r>`: Token reduction ratio (float values: 0.1, 0.15, 0.2, etc.)

Given H × W input tokens, this reduces to (H - r×H) × (W - r×W) tokens.

**Example:**
```bash
python3 test_detection.py -s 100 # baseline
python3 test_detection.py -s 100 -m cume -l 20 -r 0.1
python3 test_detection.py -s 100 -m tome -l 20 -r 0.1
python3 test_detection.py -s 100 -m expedite -l 20 -r 0.1
```