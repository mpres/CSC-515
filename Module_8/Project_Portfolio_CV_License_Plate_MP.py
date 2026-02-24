"""
License Plate Detection and OCR Pipeline
=========================================

"""
import cv2
import numpy as np
import pytesseract
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
CASCADE_PLATE_RUS  = "/usr/local/lib/python3.12/dist-packages/cv2/data/haarcascade_license_plate_rus_16stages.xml"
CASCADE_PLATE_NUM  = "/usr/local/lib/python3.12/dist-packages/cv2/data/haarcascade_russian_plate_number.xml"
