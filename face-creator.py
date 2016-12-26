import cv2
import os
import numpy as np
from firebase import firebase
from PIL import Image

firebase = firebase.FirebaseApplication('https://test-new-d1982.firebaseio.com/', None)
recognizer = cv2.face.createLBPHFaceRecognizer()
path = 'dataSet'
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)
id = raw_input('Enter user id:')
name = raw_input('Enter user name:')
sampleNum = 0

def getImagesWithID(path):
    data = {'id':id, 'name': name}
    result = firebase.post('/users/', data)
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L');
        faceNp = np.array(faceImg, 'uint8')
        ID = int(os.path.split(imagePath)[-1].split('-')[1])
        faces.append(faceNp)
        print ID
        IDs.append(ID)
        cv2.imshow("training", faceNp)
        cv2.waitKey(10)
    return IDs, faces

while True:
    ret, img = video_capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        sampleNum = sampleNum + 1
        cv2.imwrite("dataSet/User-" + str(id) + "-" + str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x,y), (x+w,y+h),( 0, 0, 255), 2)
        cv2.waitKey(100)

    cv2.imshow('Face', img)
    cv2.waitKey(1)
    if(sampleNum > 20):
        Ids, faces = getImagesWithID(path)
        recognizer.train(faces, np.array(Ids))
        recognizer.save('recognizer/trainingData.xml')
        break

video_capture.release()
cv2.destroyAllWindows()
