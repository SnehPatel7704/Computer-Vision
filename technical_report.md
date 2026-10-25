# Technical Report: Computer Vision Assignment

## 1. Objective

The goal of this assignment is to implement core Computer Vision techniques from image restoration, segmentation, feature extraction, object recognition and motion estimation. The work follows the requirements of Units 3, 4 and 5 using Python with OpenCV, NumPy, Matplotlib, scikit-image and Ultralytics YOLO.

All implementation and result presentation is contained in `Assignment_3_Computer_Vision.ipynb`; there is no separate Python pipeline file. Run the notebook from top to bottom.

## 2. Input data

A personal photograph supplied for this assignment was used as the main image input. The HEIF source was converted to PNG for OpenCV compatibility. The feature-matching pair is derived from the same photograph by resizing it; this demonstrates scale/rotation robustness but is not a second independently captured photograph.

- Supplied original photograph: `Sneh.heif`
- Processing copy: `data/original_photo.png` (4032 x 2268 pixels)
- Matching pair derived from the supplied photograph: `data/scene_a.png` and `data/scene_b.png`
- Motion-analysis video: `data/original_video.mp4` (supplied outdoor recording; 720 × 1280, 54 frames at approximately 29.9 fps, about 1.8 seconds)

## 3. Methods used

### 3.1 Image restoration and segmentation

- Gaussian noise and salt-and-pepper noise were added to the original photograph.
- Median filtering and Gaussian filtering were used to restore the noisy images.
- PSNR and SSIM were computed to compare restored quality.
- Thresholding was applied to generate a binary image.
- Morphological operations including erosion, dilation, opening and closing were performed.
- Canny edge detection was used to highlight object boundaries.
- Watershed segmentation was applied to separate touching objects.

### 3.2 Feature extraction and image matching

- HOG features were extracted and visualized.
- SIFT keypoints and descriptors were computed.
- Feature matching was performed between a scaled version of the supplied photo and a resized version of the same scene.
- The matching experiment was repeated after resizing and rotating one image to compare robustness.

### 3.3 Object recognition and motion estimation

- Haar cascade classifier was used for face detection.
- YOLOv8 was used for object detection.
- Frame differencing was used to detect moving regions.
- Background subtraction using MOG2 was used to separate foreground from background.
- Lucas-Kanade optical flow was used to estimate motion between consecutive frames.

The Haar cascade found one face. YOLO produced two `person` boxes: a high-confidence detection on the person (about 0.91) and a low-confidence false positive on the fence/vegetation (about 0.33). This is the observed detector error case.

## 4. Output summary

The processed results are stored in the `results/` folder. The combined contact sheet is `results/final_results_overview.png`; individual results include:

- denoised images and noise comparison
- PSNR and SSIM measurements
- binary masks and morphological processing outputs
- edge and watershed segmentation outputs
- HOG visualization
- SIFT keypoints and matching results
- object detection outputs
- motion masks and optical flow visualization

## 5. Restoration measurements

The same fixed random seed is used for both noise types, so the comparisons are reproducible within the same software environment.

| Noise | Restoration filter | PSNR (dB) | SSIM |
|---|---|---:|---:|
| Gaussian | Gaussian | 32.7408 | 0.8581 |
| Gaussian | Median | 31.9501 | 0.8182 |
| Salt-and-pepper | Gaussian | 27.6546 | 0.6334 |
| Salt-and-pepper | Median | 33.8344 | 0.9094 |

## 6. Comparison and discussion

The measurements show that Gaussian filtering performs better for Gaussian noise, while median filtering is clearly better for salt-and-pepper noise. Morphological operations alter region boundaries as expected. SIFT produced 1,278 good matches initially, 1,259 after rotation, and 1,140 after resizing, demonstrating robust matching on transformed copies. The observed YOLO false positive is noted above; Haar detected the photographed face. Frame differencing highlights temporal change, MOG2 provides a warmed-up foreground mask, and Lucas-Kanade supplies directional motion vectors.

## 7. Conclusion

The photo-based restoration, segmentation, feature extraction, and object-detection workflows were run on the supplied personal photograph. Haar detected one face and YOLO returned two object detections. SIFT produced 1,278 good matches on the initial pair, 1,259 after rotation, and 1,140 after resizing. Motion estimation was run on the supplied 54-frame clip, producing frame-difference, MOG2, and optical-flow visualizations.

The feature-matching pair is derived from one photograph rather than two independently captured views, and the video clip is short (about 1.8 seconds). 

## 8. Git repository and demonstration video

- Git repository: https://github.com/SnehPatel7704/Computer-Vision.git
- Project explanation video: To be uploaded to the Google Drive location provided by the instructor.
- The supplied motion video is short (54 frames, approximately 1.8 seconds).
