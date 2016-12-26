import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)
rec = cv2.face.createLBPHFaceRecognizer()
rec.load("recognizer/trainingData.xml")
id = 0
name = "undefinded"
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, img = video_capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h),(0, 0, 255), 2)
        id, conf = rec.predict(gray[y:y+h, x:x+w])
        if (id == 1):
            name = "Jerry Wang"
        elif(id == 2):
            name = "Seafood"
        elif(id == 3):
            name = "Lin"
	else:
            name = "Undefinded"
	
    	# cv2.putText(img, , , font, (0, 255, 0))
        cv2.putText(img, str(name),(x, y + h), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Face', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
