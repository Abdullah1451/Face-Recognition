import cv2 as cv
import numpy as np

from rescale import rescaleFrame

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['Abdullah', 'Abid', 'Aisha', 'Anas', 'Ashhar', 'Azhar', 'Izaan', 'Nishat']

#features = np.load('features.npy')
#labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')


img = cv.imread('D:\Faces\\train\\Nishat\IMG_20160403_222050.jpg')


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

i=0
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

for(x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+h]
    

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')
    i=i+100
    cv.putText(img, str(people[label]), (i, i), cv.FONT_HERSHEY_COMPLEX, 5.0, (0,255,0),2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)



img = rescaleFrame(img,0.30)
cv.imshow('detected', img)
cv.waitKey(0)