from collections import deque
from cv2 import CONTOURS_MATCH_I1
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

#Se definen intervalos de acuerdo a la luz
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(0)

while True:
    (grabbed, frame) = camera.read()

    if args.get("video") and not grabbed:
        break

    #frame = imutils.resize(frame, width=600)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

   
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea) #Aquí se sabe el tamaño en área
        ((x, y), radius) = cv2.minEnclosingCircle(c) #Aquí se obtiene el centroide en x y y
        
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
            cv2.circle(frame,(int(x), int(y)), 5, (0, 0, 255), -1)
            #Aquí iria la función del envio de coordenadas

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()