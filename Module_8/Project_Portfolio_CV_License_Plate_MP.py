"""
License Plate Detection and OCR Pipeline
=========================================
"""
import cv2
import numpy as np
import easyocr
from pathlib import Path

SCALE_FACTOR  = 1.2
MIN_NEIGHBORS = 5

# Initialize EasyOCR once — loading the model is expensive, do it outside functions
reader = easyocr.Reader(['en'])

# Paths for Models
CASCADE_PLATE_RUS = "Models/haarcascade_license_plate_rus_16stages.xml"
CASCADE_PLATE_NUM = "Models/haarcascade_russian_plate_number.xml"

# Load Cascade Models
cascade_rus = cv2.CascadeClassifier(CASCADE_PLATE_RUS)
cascade_num = cv2.CascadeClassifier(CASCADE_PLATE_NUM)

# Images
IMAGES = {
    "non_russian_multi": "images/license_plates_non_russian.jpg",
    "russian_multi_far": "images/Russian_Multi_far.png",
    "russian_upclose":   "images/Russian_License_Upclose.jpg",
    "european_plate":    "images/European_plate.jpg",
}

# Check models loaded
if cascade_rus.empty() or cascade_num.empty():
    raise RuntimeError("Failed to load cascade classifiers!")

print("Cascade classifiers loaded")

# Create output directory
OUTPUT_DIR = Path(__file__).parent / "Outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def detect_license_plate(img, model=cascade_rus, scale_factor=SCALE_FACTOR, min_neighbors=MIN_NEIGHBORS):
    """Detect license plates and draw red bounding boxes on a copy of the image."""
    detection_img = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Gray Image", img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plates = model.detectMultiScale(
        img_gray,
        flags=cv2.CASCADE_SCALE_IMAGE,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors
    )

    for (x, y, w, h) in plates:
        cv2.rectangle(detection_img, (x, y), (x + w, y + h), (0, 0, 255), 3)

    cv2.imshow("Detection Result", detection_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detection_img, plates


def extract_roi(img_color, plates, pad=6):
    """
    Extract padded color and grayscale ROI crops for each detected plate.
    Returns a list of dicts with keys: plate_idx, bbox, roi_color, roi_gray.
    """
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
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


def deskew_and_align(roi_gray):
    """
    Detect tilt angle via minAreaRect and rotate to horizontal.
    Scale to standard plate size for OCR.
    """
    _, thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    angle = 0.0
    if contours:
        all_pts = np.vstack(contours)
        rect    = cv2.minAreaRect(all_pts)
        angle   = rect[2]
        if angle < -45:
            angle += 90

    h, w = roi_gray.shape
    M       = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(roi_gray, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    scaled = cv2.resize(rotated, (640, 160), interpolation=cv2.INTER_LANCZOS4)
    return scaled, angle


def preprocess_gray(gray):
    """
    Bilateral filter + CLAHE for noise suppression and contrast enhancement.
    """
    denoised = cv2.bilateralFilter(gray, d=11, sigmaColor=75, sigmaSpace=75)
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return enhanced


def ocr_plate(roi_processed):
    """
    Binarise with adaptive thresholding then run EasyOCR for character recognition.
    """
    # Sharpen
    kernel    = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(roi_processed, -1, kernel)

    # Adaptive threshold — handles uneven lighting better than Otsu
    binary = cv2.adaptiveThreshold(
        sharpened,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=15,
        C=9
    )

    # EasyOCR expects a 3-channel image
    binary_3ch = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    results = reader.readtext(
        binary_3ch,
        detail=0,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )

    text = ''.join(results).strip()
    return text, binary


def process_image(img_path_key):
    """Run the full pipeline for a given image key."""
    img_path  = Path(IMAGES[img_path_key])
    img_color = cv2.imread(str(img_path))

    if img_color is None:
        print(f"ERROR: Could not read {img_path}")
        return

    print(f"\n{'='*50}")
    print(f"Processing: {img_path_key}")
    print(f"{'='*50}")

    detection_img, plates = detect_license_plate(img_color)
    print(f"  Plates detected: {len(plates)}")

    extract_rois = extract_roi(img_color, plates)

    for roi in extract_rois:
        roi_gray = roi['roi_gray']

        cv2.imshow(f"ROI #{roi['plate_idx']}", roi_gray)
        cv2.waitKey(0)

        # Preprocess → deskew → OCR
        processed = preprocess_gray(roi_gray)
        aligned, angle = deskew_and_align(processed)

        cv2.imshow("Aligned", aligned)
        cv2.waitKey(0)

        text, binary = ocr_plate(aligned)

        cv2.imshow("Binary", binary)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"  Plate #{roi['plate_idx']} | angle={angle:.1f}° | OCR: '{text}'")


# ── Run pipeline for each image ───────────────────────────────────────────────
process_image("russian_upclose")
process_image("russian_multi_far")
process_image("european_plate")