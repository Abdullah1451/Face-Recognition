import cv2 as cv
import numpy as np


img = cv.imread('photos/college.jpg')
def thresholding_img(img):

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #Adaptive Threshoding
    adaptive_threshold = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
    return adaptive_threshold#cv.imshow('Adaptive', adaptive_threshold)


#cv.waitKey(0)
