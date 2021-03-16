import cv2 as cv

def rescaleImg(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img = rescaleImg(cv.imread('photos/manyPeeps.jpg'), scale=.75)
cv.imshow('original img', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('gray', gray)

haarCascade = cv.CascadeClassifier('haar_face.xml')

faces = haarCascade.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 3, minSize=(100, 100))


for (x, y, w, h) in faces:
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)
    facesLocation = img[y:y + h, x:x + w]
    cv.imshow(str(w) + str(h) + '_faces.jpg', facesLocation)

cv.imshow('detected faces', img)
print(f'number of faces found = {len(faces)}')

cv.waitKey(0)
