import cv2
import numpy as np

# --- Settings ---
IMG_WIDTH, IMG_HEIGHT = 500, 500
BACKGROUND_COLOR = (255, 0, 0)   # Blue in BGR format (OpenCV uses BGR)


image = np.full((IMG_HEIGHT, IMG_WIDTH, 3), BACKGROUND_COLOR, dtype=np.uint8)

# Square settings
SQUARE_TOP_LEFT = (50, 50)
SQUARE_BOTTOM_RIGHT = (150, 150)
SQUARE_COLOR = (0, 255, 0)       # Green square
THICKNESS = -1                    # -1 = filled

# Circle settings
CENTER = (250, 250)
RADIUS = 50
CIRCLE_COLOR = (0, 0, 255)       # Green square



# --- Draw filled square ---
cv2.rectangle(image, SQUARE_TOP_LEFT, SQUARE_BOTTOM_RIGHT, SQUARE_COLOR, THICKNESS)

# --- Draw filled --- circle
cv2.circle(image, CENTER, RADIUS, CIRCLE_COLOR, THICKNESS)


# --- Add edge detections ----
# 1. convert image to gray
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#. Use canny
canny_edges = cv2.Canny(gray, threshold1=50, threshold2=150)

#.# --- Sobel ---
# Calculates the gradient (rate of change) in X and Y directions
# then combines them to find edge strength
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # horizontal edges
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # vertical edges
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
sobel_edges = np.uint8(np.clip(sobel_combined, 0, 255))

# --- Laplacian ---
# Detects edges by looking for rapid changes in all directions at once
laplacian_edges = cv2.Laplacian(gray, cv2.CV_64F)
laplacian_edges = np.uint8(np.clip(np.abs(laplacian_edges), 0, 255))

# ─────────────────────────────────────────────
# GROUND TRUTH
# Draw just the outlines of our shapes onto a
# blank black image — this is our "perfect" edge map
# ─────────────────────────────────────────────
def make_ground_truth():
    gt = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    cv2.rectangle(gt, SQUARE_TOP_LEFT, SQUARE_BOTTOM_RIGHT, 255, 1)  # outline only
    cv2.circle(gt, CENTER, RADIUS, 255, 1)                           # outline only

    # Dilate slightly so we allow 1-2px tolerance in detection
    kernel = np.ones((3, 3), np.uint8)
    gt = cv2.dilate(gt, kernel, iterations=1)
    return gt


def evaluate(detected, ground_truth):
    # Convert to binary (True/False) for comparison
    det = detected > 0
    gt  = ground_truth > 0

    TP = np.sum(det & gt)   # correctly detected edges
    FP = np.sum(det & ~gt)  # detected something that isn't an edge
    FN = np.sum(~det & gt)  # missed a real edge

    precision = TP / (TP + FP + 1e-8)  # 1e-8 avoids division by zero
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return round(precision, 3), round(recall, 3), round(f1, 3)


# --- Display the image ---
# --- Display all results ---
cv2.imshow("Original",          image)
cv2.waitKey(0)        # Wait until a key is pressed
cv2.imshow("Grayscale",         gray)
cv2.waitKey(0)        # Wait until a key is pressed
cv2.imshow("Canny Edges",       canny_edges)
cv2.waitKey(0)        # Wait until a key is pressed
cv2.imshow("Sobel Edges",       sobel_edges)
cv2.waitKey(0)        # Wait until a key is pressed
cv2.imshow("Laplacian Edges",   laplacian_edges)
cv2.waitKey(0)        # Wait until a key is pressed
cv2.destroyAllWindows()

