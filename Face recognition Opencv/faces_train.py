import cv2 as cv
import os
import numpy as np

people = ['Abdullah', 'Abid', 'Aisha', 'Anas', 'Ashhar', 'Azhar', 'Izaan', 'Nishat']

DIR = r'D:\\Faces\\train'


haar_cascade = cv.CascadeClassifier('Face recognition Opencv\haar_face.xml')

features = []
labels = []


def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)
        

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
           # print(img_path)
            #print(f'Img array : {img_array}')
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            #print(f'Faces rect : {faces_rect}')
            for(x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                #print(f'faces Roi : {faces_roi}')
                features.append(faces_roi)
               # print(f'Features : {features}')
                labels.append(label)
                #print(f'\nlabels : {labels}')


create_train()
print("Training Done......................")

features = np.array(features, dtype='object')
labels = np.array(labels)


face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')

np.save('features.npy', features)
np.save('labels.npy', labels)



#print(f'Length of the features = {len(features)}')
#print(f'Length of the labels = {len(labels)}')