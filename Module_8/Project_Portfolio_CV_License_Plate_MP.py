"""
License Plate Detection and OCR Pipeline
=========================================

"""
import cv2
import numpy as np
import pytesseract
from pathlib import Path

SCALE_FACTOR = 1.25
MIN_NEIGHBORS = 3


# Paths for Models
CASCADE_PLATE_RUS  = "Models/haarcascade_russian_plate_number.xml"
CASCADE_PLATE_NUM  = "Models/haarcascasde_license_plate_rus_16stages.xml"

#  Load Cascades Models
cascade_rus  = cv2.CascadeClassifier(CASCADE_PLATE_RUS)
cascade_num  = cv2.CascadeClassifier(CASCADE_PLATE_NUM)

# get Images
IMAGES = {
    "non_russian_multi":   "images/license_plates_non_russian.jpg",
    "russian_multi_far":   "images/Russian_Multi_far.png",
    "russian_upclose":     "images/Russian_License_Upclose.jpg",
    "european_plate":     "images/European_plate.jpg",
}


#check models
if cascade_rus.empty() or cascade_num.empty():
    raise RuntimeError("Failed to load cascade classifiers!")

print("Cascade classifiers loaded")

# create output file


OUTPUT_DIR = Path("Outputs/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def detect_license_plate(img, model=cascade_rus, scale_factor=SCALE_FACTOR, min_Neighbors=MIN_NEIGHBORS):
    ''' get image plates'''
    detection_img = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Show gray image
    cv2.imshow("Gray Image",img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plates = model.detectMultiScale(img_gray, flags=cv2.CASCADE_SCALE_IMAGE, scaleFactor=scale_factor,
                                          minNeighbors=min_Neighbors)
    for (x, y, w, h) in plates:
        cv2.rectangle(detection_img, (x, y), (x + w, y + h), (0, 0, 255), 3)

    cv2.imshow("Result_Image", detection_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detection_img, plates

def extract_roi(img_color, gray, plates, pad=6):
    """
    Extract padded color and grayscale ROI crops for each detected plate.
    Returns a list of dicts with keys: plate_idx, bbox, roi_color, roi_gray.
    """
    img_height, img_width = gray.shape[:2]
    rois = []

    for i, (x, y, w, h) in enumerate(plates):
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_width,  x + w + pad)
        y2 = min(img_height, y + h + pad)

        rois.append({
            "plate_idx" : i + 1,
            "bbox"      : (x, y, w, h),
            "roi_color" : img_color[y1:y2, x1:x2],
            "roi_gray"  : gray[y1:y2, x1:x2],
        })

    return rois




#Get basic image

img_path = Path(IMAGES["russian_upclose"])
img_color = cv2.imread(img_path)

if img_color is None:
        print(f"  ERROR: Could not read {img_path}")
else:
    detection_img, plates = detect_license_plate(img_color)



#Get Russian car plates faraway

img_path = Path(IMAGES["russian_multi_far"])
img_color = cv2.imread(img_path)

if img_color is None:
        print(f"  ERROR: Could not read {img_path}")
else:
    detection_img, coordinates = detect_license_plate(img_color)
    print("coordinates", coordinates)


#european_plate

img_path = Path(IMAGES["european_plate"])
img_color = cv2.imread(img_path)

if img_color is None:
        print(f"  ERROR: Could not read {img_path}")
else:
    detection_img, coordinates = detect_license_plate(img_color)
    print("coordinates", coordinates)