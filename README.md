# Face, Eyes, and Smile Detection Project

## Overview

This project detects faces, eyes, and smiles in images using OpenCV's Haar cascade classifiers. It evaluates the performance of the detection process by calculating confusion matrices and accuracy scores for each detection task (eyes and smiles).

## Features

- Detect faces, eyes, and smiles in images.
- Assign labels based on detected features.
- Evaluate detection accuracy with confusion matrices.
- Visualize results using heatmaps.

## Requirements

- Python 3.7+
- Required Libraries:
  - OpenCV
  - Matplotlib
  - Seaborn
  - scikit-learn
- Haar cascade XML files:
  - `haarcascade_frontalface_default.xml`
  - `haarcascade_eye_tree_eyeglasses.xml`
  - `haarcascade_smile.xml`

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the Haar cascade XML files are present in the project directory or provide their absolute paths in the script.

## Dataset

- Place your image dataset in a folder named `159.people` (or update the script with your folder name).
- The image filenames should start with a label number representing:
  - `0`: No features (none).
  - `1`: Eyes detected.
  - `2`: Smiles detected.

## Usage

1. Run the `main.py` script:

   ```bash
   python main.py
   ```

2. The script will:

   - Load images and extract labels from filenames.
   - Detect faces, eyes, and smiles using Haar cascades.
   - Generate confusion matrices and calculate accuracy scores for eyes and smiles detection.
   - Display heatmaps of the confusion matrices.

## Results

The script outputs:

- Confusion matrices for eyes and smiles detection.
- Accuracy scores displayed in the heatmap plots.

## Folder Structure

```
project-root/
|
|-- main.py                 # Main script for detection and evaluation
|-- 159.people/             # Folder containing the image dataset
|-- haarcascade_frontalface_default.xml
|-- haarcascade_eye_tree_eyeglasses.xml
|-- haarcascade_smile.xml
```

## Example Output

- Confusion Matrix for Eyes Detection:

- Confusion Matrix for Smiles Detection:

##
