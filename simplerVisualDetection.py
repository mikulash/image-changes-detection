import cv2
import numpy as np
import os

videos = ['./edited/1.mp4', './edited/Jur_OK_1.mp4', './edited/kostka2.mp4']
cap = cv2.VideoCapture(videos[0])

if not cap.isOpened():
    print("cannot read video input")
    exit()

# video metadata
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, cFrame = cap.read()
    if not ret:
        break
    cv2.imshow("frame", cFrame)
    cv2.waitKey(50)

cv2.destroyAllWindows()
cap.release()
