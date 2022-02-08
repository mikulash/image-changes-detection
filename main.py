import cv2
import numpy as np


# functions
def getColorAndPosition(event, x, y, flags, param):
    global detectedColor
    if event == cv2.EVENT_LBUTTONDOWN:
        print("position: ", x, y)
        print("color BGR: ", frame[y, x])
        detectedColor = frame[y, x]
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

# get color to detect
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

# get range of acceptable colors
lowColor = np.array([0, 100, 100])
upperColor = np.array([0, 255, 255])
lowColor[0] = hsv_detect[0] - 10
upperColor[0] = hsv_detect[0] + 10
print("lowColor", lowColor)
print("upperColor", upperColor)

# apply color range
ret, frame = cap.read()
frame = cv2.resize(frame, (400, 300))

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lowColor, upperColor)
prevMasked = cv2.bitwise_and(frame, frame, mask=mask)
# cv2.imshow("mask", frame)
pFrame = frame
print("LOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOP")
while True:
    ret, cFrame = cap.read()
    if not ret:
        break
    cFrame = cv2.resize(cFrame, (400, 300))
    hsv = cv2.cvtColor(cFrame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lowColor, upperColor)
    masked = cv2.bitwise_and(cFrame, cFrame, mask=mask)
    # detect errors here!!!
    # todo filter noise!! https://www.youtube.com/watch?v=_aTC-Rc4Io0&list=WL&index=48&t=320s
    diff = cv2.absdiff(prevMasked, masked)
    prevMasked = masked
    h, s, gray = cv2.split(diff)
    cv2.imshow("gray manually", diff)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sumContours = 0
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        sumContours += cv2.contourArea(contour)
        if cv2.contourArea(contour) < 900:
            continue
    print(sumContours)
    tolerovano = 500
    if sumContours > tolerovano:
        cv2.putText(cFrame, "Status: Error", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)
    #TODO function for averaging contours past frames
    cv2.drawContours(cFrame, contours, -1, (0, 0, 255), 2)
    cv2.imshow("masked", masked)
    cv2.imshow("frame", cFrame)
    cv2.waitKey(40)
    pFrame = cFrame

cv2.destroyAllWindows()
cap.release()
