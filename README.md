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

- `src/assignment_pipeline.py` – complete implementation
- `data/` – generated input photograph and video
- `results/` – all required processed outputs
- `technical_report.md` – concise technical documentation

## Prerequisites

- Python 3.10+
- OpenCV
- NumPy
- Matplotlib
- scikit-image
- Ultralytics YOLO

## Run the assignment

```bash
cd /Users/snehpatel/MIT/Git/Computer-Vision
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/assignment_pipeline.py
```

The script will generate the original inputs and save all processed outputs into the `results/` folder.

## Git repository

- GitHub: https://github.com/SnehPatel7704/Computer-Vision.git

## Notes

- Large raw video files are intentionally avoided in Git and kept as generated local assets for demonstration.
- For a formal submission, upload the generated video and final documentation to the Google Drive location provided by the instructor.
