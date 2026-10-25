# Computer Vision Assignment 3: Image Restoration, Feature Extraction, Object Recognition and Motion Estimation

This project implements the required practical coding assignment covering Units 3, 4 and 5 of the Computer Vision course.

## Assignment tasks completed

1. Image restoration and segmentation
   - Gaussian noise and salt-and-pepper noise injection
   - Median filtering and Gaussian filtering restoration
   - PSNR and SSIM comparison
   - Thresholding, erosion, dilation, opening, closing
   - Canny edge detection
   - Watershed segmentation

2. Feature extraction and image matching
   - HOG feature visualization
   - SIFT keypoint detection and descriptor extraction
   - Feature matching between similar image pairs
   - Matching comparison after rotation and resizing

3. Object recognition and motion estimation
   - Viola-Jones Haar cascade face detection
   - YOLO object detection
   - Frame differencing
   - Background subtraction using MOG2
   - Lucas-Kanade optical flow

## Repository structure

- `Assignment_3_Computer_Vision.ipynb` – complete processing implementation and step-by-step interactive report, locally validated and compatible with Colab
- `data/` – the supplied photograph (converted to PNG), transformed photo pair, and video input
- `results/` – all required processed outputs
- `technical_report.md` – concise technical documentation

## Prerequisites

- Python 3.10+
- Internet access for notebook package installation and model download

## Run the assignment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "opencv-python<5" scikit-image matplotlib pandas ultralytics jupyter
jupyter notebook Assignment_3_Computer_Vision.ipynb
```

Run the notebook cells from top to bottom. Its first code cell installs packages directly; no requirements file is used. In Colab, open the notebook from the GitHub link below. The setup cell clones this repository and uses the files in `data/` as inputs.

The supplied `Sneh.heif` photograph is represented by `data/original_photo.png` and used for restoration, segmentation, features, and object detection. The notebook leaves the original input unchanged and derives a transformed pair for feature matching. It saves processed outputs into `results/`.

The supplied `data/original_video.mp4` is used for motion analysis (720 × 1280, 54 frames at approximately 29.9 fps). The notebook validates and uses the file; it does not synthesize a substitute.

## Notebook / Colab

Open the notebook in Jupyter, or launch it directly in Colab:

- [Open Assignment 3 in Google Colab](https://colab.research.google.com/github/SnehPatel7704/Computer-Vision/blob/main/Assignment_3_Computer_Vision.ipynb)

The notebook clones this repository and reads the input photo/video from `data/`. It saves generated outputs in `results/`.
For the clone step to work, push the notebook and required `data/` inputs to GitHub first; unauthenticated Colab cloning requires the repository to be publicly accessible. A public repository makes those photo/video files visible to anyone, so upload only media you consent to share.

## Git repository

- GitHub: https://github.com/SnehPatel7704/Computer-Vision.git
