import cv2 as cv
import numpy as np

blank = np.zeros((500, 500, 3), dtype='uint8')

cv.rectangle(blank, (0,0), (250,500), (0,0,255), thickness=cv.FILLED)

cv.imshow('rec',blank)

cv.waitKey(0)

