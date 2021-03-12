import cv2 as cv

faceCascade = cv.CascadeClassifier('haar_face.xml')

cap = cv.VideoCapture(0)

while True:
    _, img = cap.read()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)

    cv.imshow('img', img)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()