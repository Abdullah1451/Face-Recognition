import cv2 as cv
from rescale import rescaleFrame

capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
     

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('Video', gray)

    if cv.waitKey(1) & 0xFF==ord('c'):
        cv.imwrite('asddas.jpg', gray)
        print('adasdadadaddasdasdasdsadbsadnbdasbdbaksabdkdkasdbkasbdjbkasjdksabfkasbfjabjjjasjfasbfjasbjabfasjhfabsfa')
    if cv.waitKey(1) & 0xFF==ord('d'):
        break    

capture.realease()
cv.destroyAllWindows()