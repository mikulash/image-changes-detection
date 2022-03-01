import cv2
import numpy as np


# functions
def alterImage(frame, height=500, width=400):
    frame = cv2.resize(frame, (height, width))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame, hsv, gray


def getColorAndPositionFromUserCallback(event, x, y, flags, param):
    global detectedColor, position
    if event == cv2.EVENT_LBUTTONDOWN:
        print("position: ", x, y)
        print("color BGR: ", frame[y, x])
        detectedColor = frame[y, x]
        position = (x, y)


def getMaskFromContours(grayImage, position):
    obr = grayImage.copy()
    sharpeningKernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharp = cv2.filter2D(obr, -1, sharpeningKernel)
    kernel = np.ones((3, 3), np.uint8)
    blur = cv2.blur(obr, (5,5))
    blur1 = cv2.bilateralFilter(obr, 9,75,75)
    edges1 = cv2.Canny(blur, 60, 180)
    # diluted = cv2.dilate(edges, kernel, iterations=1)
    # ret, thresh = cv2.threshold(gray, 127, 255, 0)
    th3 = cv2.adaptiveThreshold(
        blur1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    cv2.imshow("thresh", th3)
    cv2.imshow("sharp", sharp)
    cv2.imshow("blurrr", blur1)
    cv2.imshow("edges", edges1)
    # contours, hierarchy = cv2.findContours(blur1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("debug2", contours)
    # eroded = cv2.erode(th3, kernel, iterations=2)
    dilated = cv2.dilate(edges1, kernel, iterations=1)
    cv2.imshow("eroded", dilated)
    contours, hierarchy = cv2.findContours(
        edges1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("contours", contours)
    # contours, hierarchy = cv2.findContours(blur1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        result = cv2.pointPolygonTest(cnt, position, True)
        print(result)
        if result > 0:
            print("nasel polygon", result)
            template = np.zeros(grayImage.shape)
            cv2.drawContours(
                template, [cnt], contourIdx=-1, color=(255, 255, 255), thickness=-1)
            return np.uint8(template)
    # in case of error of stream
    return np.ones(grayImage.shape, np.uint8)


"""main program"""
# read stream
# images = ['nohy.png', 'kostka.png']
# img = cv2.imread('./imgs/'+images[1])
# mask = getMaskFromContours(img, (430, 445))
videos = ['./edited/2.mp4', './edited/Jur_OK_1.mp4', './edited/kostka2.mp4']
cap = cv2.VideoCapture(videos[2])

if not cap.isOpened():
    print("cannot read video input")
    exit()

# video metadata
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("frame_width", frame_width)
print("frame_height", frame_height)

"""get color to detect"""
cv2.namedWindow('getColor')
cv2.setMouseCallback("getColor", getColorAndPositionFromUserCallback)
detectedColor, position = None, None

"""loop to get user input"""
while True:
    # sejmi barvu objektu, ale cykli pokud se objekt jeste neobjevil
    ret, frame = cap.read()
    frame = cv2.resize(frame, (500, 400))
    cv2.imshow("getColor", frame)
    key = cv2.waitKey(0)
    if key == 39 or key == 32:  # dalsi frame pri kliku -> nebo space
        continue
    if key == 27:
        # TODO zavirat okno po vybrani barvy
        break

"""process user input"""
detectedColor = np.uint8([[detectedColor]])
hsv_detect = cv2.cvtColor(detectedColor, cv2.COLOR_BGR2HSV)
print("hsv_detect", hsv_detect)
print("detectedColor", detectedColor)
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
frame = cv2.resize(frame, (500, 400))

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lowColor, upperColor)
prevMasked = cv2.bitwise_and(frame, frame, mask=mask)
# cv2.imshow("mask", frame)
"""MAIN LOOP"""
pFrame = frame
while True:
    ret, cFrame = cap.read()
    if not ret:
        break
    cFrame, hsv, gray = alterImage(cFrame)
    hueMask = cv2.inRange(hsv, lowColor, upperColor)
    masked = cv2.bitwise_and(cFrame, cFrame, mask=hueMask)
    contourMask = getMaskFromContours(gray, position)
    masked2 = cv2.bitwise_and(cFrame, cFrame, mask=contourMask)
    cv2.imshow("MASK2", contourMask)
    cv2.imshow("MASKED2", masked2)
    """Detect errors here!!!"""
    # cv2.imshow("quantized", quantizationOfIMG(cFrame, 2, 10))
    # todo filter noise!! https://www.youtube.com/watch?v=_aTC-Rc4Io0&list=WL&index=48&t=320s
    diff = cv2.absdiff(prevMasked, masked)
    prevMasked = masked
    h, s, grayFromHUE = cv2.split(diff)
    # cv2.imshow("gray manually", diff)
    blur = cv2.GaussianBlur(grayFromHUE, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sumContours = 0
    for contour in contours:
        sumContours += cv2.contourArea(contour)
        if cv2.contourArea(contour) < 900:
            continue
    tolerance = 500
    if sumContours > tolerance:
        cv2.putText(cFrame, "Status: Error", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)
    # TODO function for averaging contours past frames
    cv2.drawContours(cFrame, contours, -1, (0, 0, 255), 2)
    # cv2.imshow("masked", masked)
    cv2.imshow("frame", cFrame)
    cv2.waitKey(40)
    pFrame = cFrame

cv2.destroyAllWindows()
cap.release()
