import cv2 as cv

img_1 = cv.imread("img_contrast_1.jpg")
#cv.imshow("img_contrast_1.jpg", img_1)
#assert img_1 is not None
#cv.waitKey(0)



img_2 = cv.imread("img_contrast_2.jpg")
#cv.imshow("img_contrast_2.jpg", img_2)
#assert img_2 is not None
#cv.waitKey(0)


#gray scale

img_1_gray = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)
assert img_1_gray is not None
cv.imshow("img_contrast_1.jpg", img_1_gray)
cv.waitKey(0)

img_2_gray = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)
assert img_1_gray is not None
cv.imshow("img_contrast_2", img_2_gray)
cv.waitKey(0)



ret,th1 = cv.threshold(img_1_gray,127,255,cv.THRESH_BINARY)
assert th1 is not None

cv.imshow("universal threshold img 1", th1)
cv.waitKey(0)


ret,th2 = cv.threshold(img_2_gray,127,255,cv.THRESH_BINARY)
assert th1 is not None

cv.imshow("universal threshold img 2", th2)
cv.waitKey(0)


th1_adaptive = cv.adaptiveThreshold(img_1_gray,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,65,3)

th1_adaptive = cv.medianBlur(th1_adaptive,33)

cv.imshow("th1_adaptive", th1_adaptive)
assert th1_adaptive is not None
cv.waitKey(0)


th2_adaptive = cv.adaptiveThreshold(img_2_gray,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,65,3)

th2_adaptive = cv.medianBlur(th2_adaptive,33)

cv.imshow("th2_adaptive", th2_adaptive)
assert th2_adaptive is not None
cv.waitKey(0)
cv.destroyAllWindows()





