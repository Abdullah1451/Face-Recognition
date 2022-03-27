import cv2 as cv


def rescaleFrame(frame, scale=0.75):#variable scale value is propotional to size of the image
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions,interpolation=cv.INTER_AREA)


"""
#resize video
capture = cv.VideoCapture('videos/Izzan.mp4')

while True:
    isTrue, frame = capture.read()
    
    frame_resized = rescaleFrame(frame)

    cv.imshow('Video', frame)
    cv.imshow('Videore', frame_resized)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.realease()
cv.destroyAllWindows()"""