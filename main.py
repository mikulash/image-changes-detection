import cv2
import numpy as np


def getColorAndPosition(event, x, y, flags, param):
    global detectedColor
    if event == cv2.EVENT_LBUTTONDOWN:
        print("position: ", x, y)
        print("color BGR: ", frame[y, x])
        detectedColor = frame[y, x]
        minVal = np.array([0, 0, 0])
        maxVal = np.array([255, 255, 255])
        # cv2.destroyWindow("getColor")


"""main program"""
# read stream
cap = cv2.VideoCapture('./edited/2.mp4')

if not cap.isOpened():
    print("cannot read video input")
    exit()

# video metadata
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("frame_width", frame_width)
print("frame_height", frame_height)

#get color to detect
cv2.namedWindow('getColor')
cv2.setMouseCallback("getColor", getColorAndPosition)

ret, frame = cap.read()
detectedColor = None

frame = cv2.resize(frame, (400, 300))
# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) zbytecny, protoze to stejne vraci BGR color, ale zobrazi to HSV
cv2.imshow("getColor", frame)
if cv2.waitKey(0) == 27:
    # TODO zavirat okno po vybrani barvy
    pass

print("detectedCOlor", detectedColor)
detectedColor = np.uint8([[detectedColor]])
hsv_detect = cv2.cvtColor(detectedColor, cv2.COLOR_BGR2HSV)
print("hsv_detect", hsv_detect)
hsv_detect = hsv_detect[0][0]

#get range of acceptable colors
lowColor = np.array([0, 100, 100])
upperColor = np.array([0, 255, 255])
lowColor[0] = hsv_detect[0] - 10
upperColor[0] = hsv_detect[0] + 10
print("lowColor", lowColor)
print("upperColor", upperColor)

#apply color range
ret, frame = cap.read()
frame = cv2.resize(frame, (400, 300))

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lowColor, upperColor)
frame = cv2.bitwise_and(frame, frame, mask=mask)
cv2.imshow("mask", frame)
if cv2.waitKey(0) == 27:
    pass

while True:
    ret, frame = cap.read()
    if frame is None:
        break

cv2.destroyAllWindows()
cap.release()