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
    "russian_multi_far":   "images/Russian_Multi_far.png",
    "russian_upclose":     "images/Russian_License_Upclose.jpg",
}


#check models
if cascade_rus.empty() or cascade_num.empty():
    raise RuntimeError("Failed to load cascade classifiers!")

print("Cascade classifiers loaded")

# create output file




OUTPUT_DIR = Path("Outputs/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


#Get basic image

img_path = Path(IMAGES["russian_upclose"])
img_color = cv2.imread(img_path)
if img_color is None:
        print(f"  ERROR: Could not read {img_path}")

img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)


#

cv2.imshow("Image", img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

#plates = detect_plates(gray_proc, cascade_rus, scaleFactor=1.05, minNeighbors=3, minSize=(60, 20))
plates = cascade_rus.detectMultiScale(img_gray,flags=cv2.CASCADE_SCALE_IMAGE, scaleFactor=1.4, minNeighbors=5)

#cv2.imshow("Image_Plates", plates)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#print(plates)


def detect_license_plate(img, plates):
    detection_img = img.copy()
    for (x, y) in plates:
        cv2.rectangle(detection_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
    return detection_img




detection_img = img_color.copy()
for (x, y, w, h) in plates:
    cv2.rectangle(detection_img, (x, y), (x + w, y + h), (0, 0, 255), 3)


cv2.imshow("Image", detection_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#im.show('test"',detection_img)



#detect raw away plates

img_path = Path(IMAGES["russian_multi_far"])
img_color = cv2.imread(img_path)
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

cv2.imshow("Image", img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

#plates = detect_plates(gray_proc, cascade_rus, scaleFactor=1.05, minNeighbors=3, minSize=(60, 20))
plates = cascade_rus.detectMultiScale(img_gray,flags=cv2.CASCADE_SCALE_IMAGE, scaleFactor=1.4, minNeighbors=5)

detection_img = img_color.copy()
for (x, y, w, h) in plates:
    cv2.rectangle(detection_img, (x, y), (x + w, y + h), (0, 0, 255), 3)

cv2.imshow("Image", detection_img)
cv2.waitKey(0)
cv2.destroyAllWindows()




# Detect non-russian plats
img_path = Path(IMAGES["non_russian_multi"])
img_color = cv2.imread(img_path)
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

cv2.imshow("Image", img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

#plates = detect_plates(gray_proc, cascade_rus, scaleFactor=1.05, minNeighbors=3, minSize=(60, 20))
plates = cascade_rus.detectMultiScale(img_gray,flags=cv2.CASCADE_SCALE_IMAGE, scaleFactor=1.4, minNeighbors=3)

#if len(plates) > 0:
#    plates = cascade_r.detectMultiScale(img_gray, flags=cv2.CASCADE_SCALE_IMAGE, scaleFactor=1.4, minNeighbors=5)



detection_img = img_color.copy()
for (x, y, w, h) in plates:
    cv2.rectangle(detection_img, (x, y), (x + w, y + h), (0, 0, 255), 3)

cv2.imshow("Image", detection_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


plates = cascade_num.detectMultiScale(img_gray, scaleFactor=1.01, minNeighbors=4)


detection_img = img_color.copy()

for (x, y, w, h) in plates:
    cv2.rectangle(detection_img, (x, y), (x + w, y + h), (0, 0, 255), 3)

cv2.imshow("Image", detection_img)
cv2.waitKey(0)
cv2.destroyAllWindows()