import cv2 as cv

#Step 1: extract image
face_img = cv.imread("front_face.jpg")

#Step 2:Draw red box left eye
cv.rectangle(face_img,(1205,775),(1375,955),(0,0,255),thickness=2)

#Step 3:Draw red box right eye
cv.rectangle(face_img,(1615,775),(1795,955),(0,0,255),thickness=2)

# Step 4. Draw circle around the face
cv.circle(face_img,(1500,850),700,(0,255,0),thickness=2)

# Step 5. Add tag
cv.putText(
    face_img,
    text="this is me",
    org=(1300, 600),                 # (x, y) position
    fontFace=cv.FONT_HERSHEY_SIMPLEX,
    fontScale=3,
    color=(255, 25, 0),            # BGR (Green)
    thickness=2,
    lineType=cv.LINE_AA
)

#Step 6: show image with red rectangle
cv.imshow("Image with Red box", face_img)
cv.waitKey(0)
cv.destroyAllWindows()
