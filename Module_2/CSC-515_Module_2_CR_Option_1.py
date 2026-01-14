import cv2 as cv

img = cv.imread("dog.jpg")

b, g, r = cv.split(img)

cv.imwrite('bluedog.png',b)

img_merged = cv.merge([b, g, r])

cv.imwrite("merged.png",img_merged)

img_r_g_swap = cv.merge([b, r, g])

cv.imwrite('dog_r_g_swap.png',img_r_g_swap)
