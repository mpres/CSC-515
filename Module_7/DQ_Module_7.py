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



# --- Display the image ---
cv2.imshow("Filled Square", image)
cv2.waitKey(0)        # Wait until a key is pressed
cv2.destroyAllWindows()

