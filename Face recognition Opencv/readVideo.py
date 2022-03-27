import cv2 as cv
from thresholding import thresholding_img
from face_recognition import video_face_recognition
from rescale import rescaleFrame

capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
     

    adaptive_thresholding = video_face_recognition(frame)
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    img = rescaleFrame(adaptive_thresholding, 1.80)
    cv.imshow('Video', img)

    if cv.waitKey(1) & 0xFF==ord('d'):
        break    

capture.realease()
cv.destroyAllWindows()