import cv2 as cv
import numpy as np

# --- Settings ---
IMG_WIDTH, IMG_HEIGHT = 500, 500
BACKGROUND_COLOR = (255, 0, 0)   # Blue in BGR format (OpenCV uses BGR)


image = np.full((IMG_HEIGHT, IMG_WIDTH, 3), BACKGROUND_COLOR, dtype=np.uint8)