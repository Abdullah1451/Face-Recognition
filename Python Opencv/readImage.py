import cv2 as cv

#importing rescaleFrame 
from rescale import rescaleFrame

#reading image
img = cv.imread('photos/cat.jpg')

#calling rescaleFrame() to resize the image
img_resized = rescaleFrame(img, 0.25)

#original image
cv.imshow('Cat', img)
#resized image
cv.imshow('Catre', img_resized)

cv.waitKey(0)