"""
License Plate Detection and OCR Pipeline
=========================================

"""
import cv2
import numpy as np
import pytesseract
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
CASCADE_PLATE_RUS  = "Models/haarcascade_russian_plate_number.xml"
CASCADE_PLATE_NUM  = "Models/haarcascasde_license_plate_rus_16stages.xml"
