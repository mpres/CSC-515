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


# Save function for easy comparison
def save_and_show(image, name):
    cv2.imwrite(os.path.join(output_dir, f"{name}.jpg"), image)
    cv2.imshow(name, image)
    cv2.waitKey(0)


save_and_show(img_gray, "1_original")

# Step 1: Noise Reduction (Gaussian Blur)
# Removes noise while preserving edges better than other blur methods
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
save_and_show(img_blur, "2_gaussian_blur")

# Step 2: CLAHE (Contrast Enhancement)
# This is GOOD - enhances ridge/valley contrast
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
img_clahe = clahe.apply(img_blur)
save_and_show(img_clahe, "3_clahe")

# Step 3: Binarization (Adaptive Thresholding)
# Better than global thresholding for uneven lighting
img_binary = cv2.adaptiveThreshold(
    img_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)
save_and_show(img_binary, "4_adaptive_threshold")

# Step 4: Morphological Operations
# Opening (erosion then dilation) - removes small noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
img_opening = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel, iterations=1)
save_and_show(img_opening, "5_morphological_opening")

# Step 5: Closing (dilation then erosion) - fills small gaps in ridges
img_closing = cv2.morphologyEx(img_opening, cv2.MORPH_CLOSE, kernel, iterations=1)
save_and_show(img_closing, "6_morphological_closing")

# Step 6: Ridge Thinning (Skeletonization) - optional but useful
# Makes ridges 1 pixel wide for minutiae detection
from skimage.morphology import skeletonize

img_skeleton = skeletonize(img_closing // 255).astype(np.uint8) * 255
save_and_show(img_skeleton, "7_skeleton")


# Alternative: Frequency Domain Enhancement (Gabor Filters)
# Very effective for fingerprints - enhances ridge patterns
def apply_gabor_filter(img, ksize=31):
    """Apply Gabor filter bank for ridge enhancement"""
    filters = []
    num_filters = 8
    for theta in np.arange(0, np.pi, np.pi / num_filters):
        kernel = cv2.getGaborKernel(
            (ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F
        )
        filters.append(kernel)

    # Apply all filters and take maximum response
    filtered_imgs = [cv2.filter2D(img, cv2.CV_8UC1, k) for k in filters]
    return np.maximum.reduce(filtered_imgs)


img_gabor = apply_gabor_filter(img_clahe)
save_and_show(img_gabor, "8_gabor_enhanced")

cv2.destroyAllWindows()

print(f"All processed images saved to: {output_dir}")