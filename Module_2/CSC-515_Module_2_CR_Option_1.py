import cv2 as cv

#Step 1, extract color channels
img = cv.imread("dog.jpg")

b, g, r = cv.split(img)

#Step 2, remerge the color channels into one image 

img_merged = cv.merge([b, g, r])

cv.imwrite("merged.png",img_merged)

#Step 3, swap the red and green channel
img_r_g_swap = cv.merge([b, r, g])

cv.imwrite('dog_r_g_swap.png',img_r_g_swap)
