import cv2 as cv

img_1 = cv.imread("img_contrast_1.jpg")
cv.imshow("img_contrast_1.jpg", img_1)
cv.waitKey(0)
cv.destroyAllWindows()

img_2 = cv.imread("img_contrast_2.jpg")
cv.imshow("img_contrast_2.jpg", img_2)
cv.waitKey(0)
cv.destroyAllWindows()


