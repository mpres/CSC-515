#Author/Date - MPresley 1/27/26
#Purpose - Get a blurry image and process using 3 filters with different parameters, for a total of 12 images
#Outline - 1. Configuration, 2. Get image, 3. Apply Blurring technique, 4

import cv2
import os


# 1. Configuration
image_path = "blurry_image.jpg"
output_dir = "blurred_results"
os.makedirs(output_dir, exist_ok=True)

# 1.b Set Kernel sizes (used in looping step)
kernel_n = [3, 5, 7]
# 1.c Set Sigma values used for Gaussian filters
sigmaValues = [0.5, 1.5]

# Get image
img = cv2.imread(image_path)

# error handling for missing file
if img is None:
    raise FileNotFoundError("Could not find image.")

#3. Apply blurring techniques
for k in kernel_n:
    # Mean Blur
    mean_blur = cv2.blur(img, (k, k))
    cv2.imwrite(f"{output_dir}/mean_{k}x{k}.jpg", mean_blur)

    # Median Blur
    median_blur = cv2.medianBlur(img, k)
    cv2.imwrite(f"{output_dir}/median_{k}x{k}.jpg", median_blur)

    # Gaussian Blurs (two sigmas)
    for Sigma in sigmaValues:
        gaussian_blur = cv2.GaussianBlur(img,(k, k),sigmaX=Sigma,sigmaY=Sigma)
        cv2.imwrite(f"{output_dir}/gaussian_{k}x{k}_sigma_{Sigma}.jpg",gaussian_blur)

print("✅ All 12 blurred images generated.")
