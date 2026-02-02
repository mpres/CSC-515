import cv2
import os


# 1. Configuration
image_path = "latent_finger_print.jpg"
output_dir = "process_finger_print_images"
os.makedirs(output_dir, exist_ok=True)