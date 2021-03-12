import cv2 as cv

img = cv.imread('photos/grupemask.jpg')
cv.imshow('original img', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('gray', gray)

haarCascade = cv.CascadeClassifier('haar_face.xml')

facesRect = haarCascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 4)

print(f'number of faces found = {len(facesRect)}')

for (x,y,w,h) in facesRect:
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)

cv.imshow('detected faces', img)

cv.waitKey(0)

