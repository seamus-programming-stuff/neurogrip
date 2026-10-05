import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

ret, frame = cap.read()

gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

cv.imshow('frame', gray)

if cv.waitKey(1) == ord('q'):
    cv.destroyAllWindows()