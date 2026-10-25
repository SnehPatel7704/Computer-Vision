# Technical Report: Computer Vision Assignment

## 1. Objective

The goal of this assignment is to implement core Computer Vision techniques from image restoration, segmentation, feature extraction, object recognition and motion estimation. The work follows the requirements of Units 3, 4 and 5 using Python with OpenCV, NumPy, Matplotlib, scikit-image and Ultralytics YOLO.

## 2. Input data

A synthetic dataset was generated for this assignment to ensure reproducibility and to avoid dependence on unavailable personal media.

- Original photograph: `data/original_photo.png`
- Pair of matching scene images: `data/scene_a.png` and `data/scene_b.png`
- Original self-recorded style video: `data/original_video.mp4`

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
- Feature matching was performed between the two scene images.
- The matching experiment was repeated after resizing and rotating one image to compare robustness.

### 3.3 Object recognition and motion estimation

- Haar cascade classifier was used for face detection.
- YOLOv8 was used for object detection.
- Frame differencing was used to detect moving regions.
- Background subtraction using MOG2 was used to separate foreground from background.
- Lucas-Kanade optical flow was used to estimate motion between consecutive frames.

## 4. Output summary

The processed results are stored in the `results/` folder and include:

- denoised images and noise comparison
- PSNR and SSIM measurements
- binary masks and morphological processing outputs
- edge and watershed segmentation outputs
- HOG visualization
- SIFT keypoints and matching results
- object detection outputs
- motion masks and optical flow visualization

## 5. Comparison and discussion

The comparison shows that median filtering performs better on salt-and-pepper noise, while Gaussian filtering is more effective for Gaussian noise. Morphological operations help refine region boundaries. HOG provides gradient-based descriptors, and SIFT remains robust for matching transformed objects. YOLO often provides more general and accurate detections than Haar cascades, although the Haar method remains useful for constrained face detection tasks. In motion analysis, frame differencing is simple but sensitive to noise, background subtraction is better for moving-object segmentation, and optical flow provides directional motion estimation between frames.

## 6. Conclusion

This assignment successfully implements the required computer vision workflows and produces a complete, reproducible solution. The codebase, generated dataset and outputs are organized neatly for academic submission and future extension.

## 7. Git repository and demonstration video

- Git repository: https://github.com/SnehPatel7704/Computer-Vision.git
- Project explanation video: To be uploaded to the Google Drive location provided by the instructor.
