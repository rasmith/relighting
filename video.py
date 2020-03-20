import cv2
import numpy as np

cap = cv2.VideoCapture('videos/IMG_0163.m4v')

if (cap.isOpened() == False):
    print("Error opening video file")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        pass
    else:
        break

