import cv2
import os

# -----------------------------
# Configuration
# -----------------------------
image_path = "blurry_image.jpg"        # <-- change this
output_dir = "blurred_results"
os.makedirs(output_dir, exist_ok=True)

# Kernel sizes
kernel_sizes = [3, 5, 7]

# Gaussian sigma values
sigma_values = [0.5, 1.5]

# -----------------------------
# Load image
# -----------------------------
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError("Could not load image. Check the path.")

# -----------------------------
# Apply blurring techniques
# -----------------------------
for k in kernel_sizes:
    # Mean Blur
    mean_blur = cv2.blur(img, (k, k))
    cv2.imwrite(f"{output_dir}/mean_{k}x{k}.jpg", mean_blur)

    # Median Blur
    median_blur = cv2.medianBlur(img, k)
    cv2.imwrite(f"{output_dir}/median_{k}x{k}.jpg", median_blur)

    # Gaussian Blurs (two sigmas)
    for sigma in sigma_values:
        gaussian_blur = cv2.GaussianBlur(
            img,
            (k, k),
            sigmaX=sigma,
            sigmaY=sigma
        )
        cv2.imwrite(
            f"{output_dir}/gaussian_{k}x{k}_sigma_{sigma}.jpg",
            gaussian_blur
        )

print("✅ All 12 blurred images generated.")
