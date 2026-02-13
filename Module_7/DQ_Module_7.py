import cv2
import numpy as np

# --- Settings ---
IMG_WIDTH, IMG_HEIGHT = 500, 500
BACKGROUND_COLOR = (255, 0, 0)   # Blue in BGR format (OpenCV uses BGR)


image = np.full((IMG_HEIGHT, IMG_WIDTH, 3), BACKGROUND_COLOR, dtype=np.uint8)



# --- Display the image ---
cv2.imshow("Filled Square", image)
cv2.waitKey(0)        # Wait until a key is pressed
cv2.destroyAllWindows()
