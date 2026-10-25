import os
import ssl
from pathlib import Path
from urllib.request import urlopen

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from skimage.feature import hog
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

HAAR_CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"


def get_haar_cascade_path() -> str:
    cascade_dir = DATA_DIR / "haarcascades"
    cascade_dir.mkdir(exist_ok=True, parents=True)
    cascade_path = cascade_dir / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        context = ssl._create_unverified_context()
        with urlopen(HAAR_CASCADE_URL, context=context) as response, open(cascade_path, "wb") as file:
            file.write(response.read())
    return str(cascade_path)


def save_image(path: Path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), image)


def generate_photo() -> Path:
    img = np.full((800, 1000, 3), (210, 220, 235), dtype=np.uint8)

    cv2.rectangle(img, (0, 600), (1000, 800), (120, 170, 110), -1)
    cv2.rectangle(img, (0, 0), (1000, 180), (255, 255, 255), -1)

    cv2.rectangle(img, (180, 260), (340, 520), (85, 92, 110), -1)
    cv2.rectangle(img, (180, 260), (340, 320), (130, 160, 230), -1)

    cv2.circle(img, (540, 260), 110, (228, 198, 165), -1)
    cv2.rectangle(img, (470, 370), (610, 560), (70, 90, 150), -1)
    cv2.rectangle(img, (485, 420), (595, 560), (80, 120, 190), -1)

    cv2.circle(img, (500, 230), 16, (0, 0, 0), -1)
    cv2.circle(img, (580, 230), 16, (0, 0, 0), -1)
    cv2.ellipse(img, (540, 270), (38, 25), 0, 0, 180, (0, 0, 0), 3)

    cv2.circle(img, (760, 330), 45, (52, 108, 207), -1)
    cv2.rectangle(img, (715, 380), (805, 470), (171, 176, 179), -1)

    cv2.putText(img, "CV Assignment", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.8, (0, 0, 0), 4, cv2.LINE_AA)

    path = DATA_DIR / "original_photo.png"
    save_image(path, img)
    return path


def generate_scene_pair() -> tuple[Path, Path]:
    img_a = np.zeros((500, 500, 3), dtype=np.uint8)
    img_a[:] = (235, 240, 245)
    cv2.rectangle(img_a, (120, 170), (330, 420), (120, 90, 50), -1)
    cv2.circle(img_a, (220, 120), 70, (52, 120, 220), -1)
    cv2.putText(img_a, "Object A", (130, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    img_b = cv2.rotate(img_a.copy(), cv2.ROTATE_90_CLOCKWISE)
    img_b = cv2.resize(img_b, None, fx=1.2, fy=1.2)

    path_a = DATA_DIR / "scene_a.png"
    path_b = DATA_DIR / "scene_b.png"
    save_image(path_a, img_a)
    save_image(path_b, img_b)
    return path_a, path_b


def generate_video() -> Path:
    video_path = DATA_DIR / "original_video.mp4"
    fps = 25
    width, height = 640, 360
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError("Failed to open video writer")

    for frame_idx in range(75):
        img = np.full((height, width, 3), (30, 60, 120), dtype=np.uint8)
        cv2.rectangle(img, (0, 230), (640, 360), (60, 120, 80), -1)
        cx = int(100 + frame_idx * 4)
        cy = 140 + int(20 * np.sin(frame_idx / 8))
        cv2.circle(img, (cx, cy), 25, (0, 0, 255), -1)
        cv2.rectangle(img, (cx - 35, cy + 30), (cx + 35, cy + 80), (255, 255, 255), -1)
        cv2.putText(img, "Moving object", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(img)

    writer.release()
    return video_path


def add_noise_and_restore(photo_path: Path) -> None:
    img = cv2.imread(str(photo_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {photo_path}")

    gaussian_noise = np.random.normal(0, 20, img.shape).astype(np.float32)
    noisy_gaussian = cv2.add(img.astype(np.float32), gaussian_noise)
    noisy_gaussian = np.clip(noisy_gaussian, 0, 255).astype(np.uint8)

    pepper_salt = img.copy()
    probability = 0.03
    flat = pepper_salt.reshape(-1, 3)
    random_mask = np.random.rand(flat.shape[0]) < probability
    flat[random_mask] = 0
    random_mask = np.random.rand(flat.shape[0]) < probability
    flat[random_mask] = 255

    restored_gaussian = cv2.GaussianBlur(noisy_gaussian, (5, 5), 1.5)
    restored_median = cv2.medianBlur(pepper_salt, 5)

    gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_gaussian = cv2.cvtColor(restored_gaussian, cv2.COLOR_BGR2GRAY)
    gray_median = cv2.cvtColor(restored_median, cv2.COLOR_BGR2GRAY)

    gaussian_psnr = psnr(gray_original, gray_gaussian)
    median_psnr = psnr(gray_original, gray_median)
    gaussian_ssim = ssim(gray_original, gray_gaussian)
    median_ssim = ssim(gray_original, gray_median)

    save_image(RESULTS_DIR / "noise_gaussian.png", noisy_gaussian)
    save_image(RESULTS_DIR / "noise_salt_pepper.png", pepper_salt)
    save_image(RESULTS_DIR / "restored_gaussian.png", restored_gaussian)
    save_image(RESULTS_DIR / "restored_median.png", restored_median)

    metrics = {
        "gaussian_psnr": gaussian_psnr,
        "median_psnr": median_psnr,
        "gaussian_ssim": gaussian_ssim,
        "median_ssim": median_ssim,
    }
    print("Restoration metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


def morphological_segmentation() -> None:
    img = cv2.imread(str(DATA_DIR / "original_photo.png"), cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)

    erosion = cv2.erode(binary, kernel, iterations=1)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(img, 50, 150)

    save_image(RESULTS_DIR / "binary_threshold.png", binary)
    save_image(RESULTS_DIR / "morph_erosion.png", erosion)
    save_image(RESULTS_DIR / "morph_dilation.png", dilation)
    save_image(RESULTS_DIR / "morph_opening.png", opening)
    save_image(RESULTS_DIR / "morph_closing.png", closing)
    save_image(RESULTS_DIR / "edge_canny.png", edges)

    watershed_source = np.full((400, 400), 255, dtype=np.uint8)
    cv2.circle(watershed_source, (120, 120), 65, 0, -1)
    cv2.circle(watershed_source, (250, 140), 70, 0, -1)
    cv2.circle(watershed_source, (190, 260), 80, 0, -1)
    cv2.circle(watershed_source, (330, 260), 60, 0, -1)
    cv2.circle(watershed_source, (100, 300), 55, 0, -1)
    dist = cv2.distanceTransform(watershed_source, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.7 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(cv2.bitwise_not(sure_fg), cv2.bitwise_not(watershed_source))
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(watershed_source, cv2.COLOR_GRAY2BGR), markers)
    watershed_result = cv2.cvtColor(watershed_source, cv2.COLOR_GRAY2BGR)
    watershed_result[markers == -1] = (0, 0, 255)
    save_image(RESULTS_DIR / "watershed_segmentation.png", watershed_result)


def feature_extraction() -> None:
    scene_a = cv2.imread(str(DATA_DIR / "scene_a.png"), cv2.IMREAD_GRAYSCALE)
    scene_b = cv2.imread(str(DATA_DIR / "scene_b.png"), cv2.IMREAD_GRAYSCALE)

    fd_a, hog_image_a = hog(scene_a, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm="L2-Hys", visualize=True)
    hog_image_a = exposure.rescale_intensity(hog_image_a, in_range=(0, 10))
    hog_vis = (hog_image_a * 255).astype(np.uint8)
    save_image(RESULTS_DIR / "hog_visualization.png", cv2.cvtColor(hog_vis, cv2.COLOR_GRAY2BGR))

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(scene_a, None)
    kp2, des2 = sift.detectAndCompute(scene_b, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    match_img = cv2.drawMatches(scene_a, kp1, scene_b, kp2, good[:50], None, flags=2)
    save_image(RESULTS_DIR / "sift_matches.png", match_img)

    rotated = cv2.rotate(scene_b.copy(), cv2.ROTATE_90_CLOCKWISE)
    resized = cv2.resize(scene_b, None, fx=1.3, fy=1.3)

    kp3, des3 = sift.detectAndCompute(rotated, None)
    kp4, des4 = sift.detectAndCompute(resized, None)
    matches_2 = bf.knnMatch(des1, des3, k=2)
    good_2 = []
    for m, n in matches_2:
        if m.distance < 0.75 * n.distance:
            good_2.append(m)
    match_img_2 = cv2.drawMatches(scene_a, kp1, rotated, kp3, good_2[:50], None, flags=2)
    save_image(RESULTS_DIR / "sift_matches_rotated.png", match_img_2)

    matches_3 = bf.knnMatch(des1, des4, k=2)
    good_3 = []
    for m, n in matches_3:
        if m.distance < 0.75 * n.distance:
            good_3.append(m)
    match_img_3 = cv2.drawMatches(scene_a, kp1, resized, kp4, good_3[:50], None, flags=2)
    save_image(RESULTS_DIR / "sift_matches_resized.png", match_img_3)

    print(f"HOG descriptor length: {len(fd_a)}")
    print(f"SIFT matches original: {len(good)}")
    print(f"SIFT matches rotated: {len(good_2)}")
    print(f"SIFT matches resized: {len(good_3)}")


def object_detection() -> None:
    img = cv2.imread(str(DATA_DIR / "original_photo.png"), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("Original photo not found")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(get_haar_cascade_path())
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    save_image(RESULTS_DIR / "haar_face_detection.png", img)
    print(f"Haar faces detected: {len(faces)}")

    model = YOLO("yolov8n.pt")
    result = model(img, conf=0.25, verbose=False)[0]
    annot = result.plot(conf=True)
    save_image(RESULTS_DIR / "yolo_detection.png", annot)
    print(f"YOLO detections: {len(result.boxes)}")


def motion_estimation() -> None:
    video_path = DATA_DIR / "original_video.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {video_path}")

    ret, prev = cap.read()
    if not ret:
        raise RuntimeError("Unable to read initial video frame")
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=40, varThreshold=25, detectShadows=True)

    optical_flow = None
    while True:
        ret, curr = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(prev_gray, curr_gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        _ = thresh
        fg_mask = bg_subtractor.apply(curr)
        _, fg_mask_thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        if corners is not None:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, corners, None)
            good_old = corners[status == 1]
            good_new = next_pts[status == 1]
            optical_flow = curr.copy()
            for old, new in zip(good_old, good_new):
                a, b = old.ravel()
                c, d = new.ravel()
                cv2.arrowedLine(optical_flow, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            break

        prev_gray = curr_gray

    cap.release()
    if optical_flow is not None:
        save_image(RESULTS_DIR / "optical_flow.png", optical_flow)

    cap = cv2.VideoCapture(str(video_path))
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    if not ret:
        raise RuntimeError("Could not read enough frames for motion comparison")

    diff_frame = cv2.absdiff(frame1, frame2)
    gray_diff = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)
    _, diff_bin = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

    save_image(RESULTS_DIR / "frame_difference.png", diff_bin)
    save_image(RESULTS_DIR / "background_subtraction.png", cv2.cvtColor(fg_mask_thresh, cv2.COLOR_GRAY2BGR))
    cap.release()


def main() -> None:
    print("Generating input assets...")
    photo_path = generate_photo()
    generate_scene_pair()
    generate_video()

    print("\nRunning image restoration...")
    add_noise_and_restore(photo_path)

    print("\nRunning morphological processing and segmentation...")
    morphological_segmentation()

    print("\nRunning feature extraction...")
    feature_extraction()

    print("\nRunning object recognition...")
    object_detection()

    print("\nRunning motion estimation...")
    motion_estimation()

    print(f"\nAll outputs saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
