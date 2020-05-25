import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap.set(3,720)
cap.set(4,1080)

def resize_frame(frame, percent = 75):
    width = int(frame.shape[1]*0.01*percent)
    height = int(frame.shape[0]*0.01*percent)
    dimension = (width,height)
    return cv2.resize(frame,dimension,interpolation = cv2.INTER_AREA)

while 1:
    ret, frame = cap.read()  # ret returns true if read is success
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    resized_frame = resize_frame(gray, 60)
    cv2.imshow('gray',resized_frame)

    cv2.imshow('Frame',frame)

    if (cv2.waitKey(20) & 0xff) == ord('q'):
        break;
cap.release()
cv2.destroyAllWindows()
