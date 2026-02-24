"""
License Plate Detection and OCR Pipeline
=========================================

"""
import cv2
import numpy as np
import pytesseract
from pathlib import Path

# Paths for Models
CASCADE_PLATE_RUS  = "Models/haarcascade_russian_plate_number.xml"
CASCADE_PLATE_NUM  = "Models/haarcascasde_license_plate_rus_16stages.xml"

#  Load Cascades Models
cascade_rus  = cv2.CascadeClassifier(CASCADE_PLATE_RUS)
cascade_num  = cv2.CascadeClassifier(CASCADE_PLATE_NUM)

# get Images
IMAGES = {
    "non_russian_multi":   "images/license_plates_non_russian.jpg",
    "russian_multi_far":   "/images/Russian_Multi_far.png",
    "russian_upclose":     "/images/Russian_Upclose.png",
}

#check models
if cascade_rus.empty() or cascade_num.empty():
    raise RuntimeError("Failed to load cascade classifiers!")

print("Cascade classifiers loaded")

# create output file

OUTPUT_DIR = Path("/Outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


#Get basic image

img_path = Path(IMAGES["russian_upclose"])
img_color = cv2.imread(img_path)

if img_color is None:
        print(f"  ERROR: Could not read {img_path}")
