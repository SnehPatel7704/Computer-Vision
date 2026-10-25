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
jupyter notebook Assignment_3_Computer_Vision.ipynb
```

Run the notebook cells from top to bottom. The supplied `Sneh.heif` photograph is represented by `data/original_photo.png` and used for restoration, segmentation, features, and object detection. The notebook leaves the original input unchanged and derives a transformed pair for feature matching. It saves processed outputs into `results/`.

The supplied `data/original_video.mp4` is used for motion analysis (720 × 1280, 54 frames at approximately 29.9 fps). The notebook validates and uses the file; it does not synthesize a substitute.

## Notebook / Colab

Open `Assignment_3_Computer_Vision.ipynb` in Jupyter or upload it to Google Colab. For Colab, make the notebook, `data/`, and `requirements.txt` available in the runtime and set the working directory to the repository root before running cells. Avoid uploading personal photos or videos to a hosted runtime unless you are comfortable sharing them with that service.

## Git repository

- GitHub: https://github.com/SnehPatel7704/Computer-Vision.git

## Notes

- Large raw videos should be stored outside Git (for example, in the instructor-provided Google Drive) and copied to `data/original_video.mp4` locally for analysis.
- For a formal submission, upload the required project explanation video and final documentation to the Google Drive location provided by the instructor.
