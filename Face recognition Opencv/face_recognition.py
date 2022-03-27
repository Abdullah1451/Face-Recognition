import cv2 as cv
import numpy as np

from rescale import rescaleFrame

haar_cascade = cv.CascadeClassifier('Face recognition Opencv\haar_face.xml')

people = ['Abdullah', 'Abid', 'Aisha', 'Anas', 'Ashhar', 'Azhar', 'Izaan', 'Nishat']

#features = np.load('features.npy')
#labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')


#img = cv.imread('D:\Faces\\train\Abdullah\\6.jpg')

def video_face_recognition(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    i = 0
    for(x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h,x:x+h]
        

        label, confidence = face_recognizer.predict(faces_roi)
        print(f'Label = {people[label]} with a confidence of {confidence}')
        
        cv.putText(img, str(people[label]), (i+100,i+100), cv.FONT_HERSHEY_COMPLEX, 2.0, (0,255,0),2)
        
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    
    return img


"""
    img = rescaleFrame(img,0.40)
    cv.imshow('detected', img)
    cv.waitKey(0)
    """