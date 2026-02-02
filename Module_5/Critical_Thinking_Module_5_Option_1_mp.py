import cv2
import numpy as np
import os

# Configuration
image_path = "latent_finger_print.jpg"
output_dir = "process_finger_print_images"
os.makedirs(output_dir, exist_ok=True)

# Load image
img = cv2.imread(image_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def save_and_show(image, name):
    cv2.imwrite(os.path.join(output_dir, f"{name}.jpg"), image)
    cv2.imshow(name, image)
    cv2.waitKey(0)

save_and_show(img_gray, "1_original")

# ========== IMPROVED NOISE REDUCTION PIPELINE ==========

# Step 1: Aggressive Noise Reduction FIRST (before CLAHE)
# Non-local Means Denoising - excellent for this type of noise
img_denoised = cv2.fastNlMeansDenoising(img_gray, None, h=12, templateWindowSize=7, searchWindowSize=21)
save_and_show(img_denoised, "2_denoised")

# Step 2: Bilateral Filter - preserves edges while smoothing
img_bilateral = cv2.bilateralFilter(img_denoised, 12, 75, 75)
save_and_show(img_bilateral, "3_bilateral")

# Step 3: NOW apply CLAHE on the cleaned image
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#img_clahe = clahe.apply(img_bilateral)
#save_and_show(img_clahe, "4_clahe")

# Step 4: Adaptive Thresholding
img_binary = cv2.adaptiveThreshold(
    img_bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)
save_and_show(img_binary, "5_adaptive_threshold")

# Step 5: Morphological Operations to remove small noise blobs
# Use a larger kernel for more aggressive noise removal
kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Opening - removes small white noise
img_opening = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel_small, iterations=1)
save_and_show(img_opening, "6_opening")

# Step 6: Find and keep only the largest connected component (the fingerprint)
# This removes all

#