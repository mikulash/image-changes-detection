import cv2
import numpy as np
import os
from statistics import mean

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub


def avgError(threshold, imageChanges):
    chyba = False
    if len(imageChanges) > 5:
        if mean(imageChanges) > threshold:
            chyba = True
            print(mean(imageChanges))
        imageChanges.pop(0)
    print(imageChanges, chyba)
    return chyba


# functions

def otevrit(image):
    kernel = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((4, 4), np.uint8)
    kernel3 = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(image, kernel)
    diluted = cv2.dilate(eroded, kernel2, iterations=1)
    eroded = cv2.erode(diluted, kernel3)
    return eroded


#
def alterImage(frame, proportions=(500, 400)):
    frame = cv2.resize(frame, proportions)
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


def getMaskFromContours(grayImage, position, clrd):
    clrd = clrd.copy()
    obr = grayImage.copy()
    sharpeningKernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(obr, -1, sharpeningKernel)
    kernel = np.ones((2, 2), np.uint8)
    bluredSharp = cv2.bilateralFilter(sharp, 9, 75, 75)
    edgesSharp = cv2.Canny(bluredSharp, 120, 255, 1)
    # eroze a pak diletace = otevreni, oddeleni od sebe blizkych objektu
    # eroded = cv2.erode(edgesSharp, kernel)
    diluted = cv2.dilate(edgesSharp, kernel, iterations=1)
    # cv2.imshow("edges", diluted)
    # ret, thresh = cv2.threshold(gray, 127, 255, 0)
    # contours, hierarchy = cv2.findContours(blur1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv2.findContours(
    #     diluted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(diluted, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(clrd, contours, -1, (0, 255, 255), 3)
    # cv2.imshow("contoured", clrd)
    template = np.zeros(grayImage.shape)
    clrd = cv2.circle(clrd, position, radius=3, color=(255, 0, 255), thickness=-1)
    if contours is not None:
        for cnt in contours:
            result = cv2.pointPolygonTest(cnt, position, True)
            # print(result)
            if result > 0:
                # debug
                cv2.drawContours(clrd, [cnt], 0, color=(0, 255, 0), thickness=1)
                # output
                cv2.drawContours(
                    template, [cnt], -1, color=(255, 255, 255), thickness=-1)
            else:
                cv2.drawContours(clrd, [cnt], contourIdx=-1, color=(0, 0, 125), thickness=-1)
    # in case of error of stream
    # cv2.imshow("CONTOURSSS", clrd)
    # TODO prekryvani contours hran ruznych hran je potreba jeste optimalizovat, mozna odecist prvni image? to by mohlo odstranit hranu podlozky
    # cv2.imshow("teamplate", template)
    return np.uint8(template)


"""main program"""
# read stream
# images = ['nohy.png', 'kostka.png']
# img = cv2.imread('./imgs/'+images[1])
# mask = getMaskFromContours(img, (430, 445))
videos = ['./edited/1.mp4', './edited/Jur_OK_1.mp4', './edited/kostka2.mp4']
cap = cv2.VideoCapture(videos[0])

# model = keras.models.load_model('./models/myModels/transferModelV2', custom_objects={'KerasLayer': hub.KerasLayer})

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
finalWidth = (200, 200)
"""loop to get user input"""
while True:
    # sejmi barvu objektu, ale cykli pokud se objekt jeste neobjevil
    ret, frame = cap.read()
    frame = cv2.resize(frame, finalWidth)
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
lowColor[0] = hsv_detect[0] - 15
upperColor[0] = hsv_detect[0] + 15
print("lowColor", lowColor)
print("upperColor", upperColor)

# apply color range
ret, frame = cap.read()
frame = cv2.bilateralFilter(frame, 15, 75, 75)  # snizit d=velikost filtru, pokud nebude stihat
frame, hsv, _ = alterImage(frame, finalWidth)

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lowColor, upperColor)
prevMasked = cv2.bitwise_and(frame, frame, mask=mask)
"""MAIN LOOP"""
pFrame = frame
imageChanges = []
while True:
    ret, cFrame = cap.read()
    if not ret:
        break
    cFrame = cv2.bilateralFilter(cFrame, 5, 75, 75)
    cFrame, hsv, gray = alterImage(cFrame, finalWidth)
    hueMask = cv2.inRange(hsv, lowColor, upperColor)
    masked = cv2.bitwise_and(cFrame, cFrame, mask=hueMask)
    # contourMask = getMaskFromContours(gray, position, cFrame)
    # masked2 = cv2.bitwise_and(cFrame, cFrame, mask=contourMask)
    """Detect errors here!!!"""
    diff = cv2.absdiff(prevMasked, masked)
    h, s, grayFromHUE = cv2.split(diff)
    cv2.imshow("gray manually", diff)
    blur = cv2.GaussianBlur(grayFromHUE, (5, 5), 0)
    cv2.imshow("gauss", blur)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sumContours = 0
    for contour in contours:
        sumContours += cv2.contourArea(contour)
        if cv2.contourArea(contour) < 900:
            continue
    tolerance = 1000
    print(sumContours)
    imageChanges.append(sumContours)
    chybaPretrvava = avgError(tolerance, imageChanges)
    if sumContours > tolerance and chybaPretrvava:
        cv2.putText(cFrame, "Status: Error", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)
        cv2.drawContours(kopie, contours, -1, (0, 0, 255), 2)
    # TODO function for averaging contours past frames
    kopie = cFrame.copy()

    cv2.imshow("masked", masked)
    cv2.imshow("kopie", kopie)
    cv2.waitKey(50)
    pFrame = cFrame
    prevMasked = masked

# new one
# ret, frame = cap.read()
# frame = cv2.bilateralFilter(frame, 15, 75, 75) #snizit d=velikost filtru, pokud nebude stihat
# frame, hsv, _ = alterImage(frame, finalWidth)
# mask = cv2.inRange(hsv, lowColor, upperColor)
# prevMasked = cv2.bitwise_and(frame, frame, mask=mask)
# pFrame = frame
# while True:
#     ret, cFrame = cap.read()
#     if not ret:
#         break
#     cFrame = cv2.bilateralFilter(cFrame, 5, 75, 75)
#     cFrame, hsv, gray = alterImage(cFrame, finalWidth)
#     hueMask = cv2.inRange(hsv, lowColor, upperColor)
#     masked = cv2.bitwise_and(cFrame, cFrame, mask=hueMask)
#     cv2.imshow("masked", masked)
#     diff = cv2.absdiff(prevMasked, masked)
#     _,_, grayFromHUE = cv2.split(diff)
#     tolerance = 1000
#     sumContours = 0
#     _, thresh = cv2.threshold(grayFromHUE, 20, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(grayFromHUE, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     for contour in contours:
#         sumContours += cv2.contourArea(contour)
#     print(sumContours)
#     if sumContours > tolerance:
#         cv2.putText(cFrame, "Status: Error", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                     1, (0, 0, 255), 3)
#
#     cv2.imshow("diffe2", diff)
#     cv2.waitKey(50)
#     pFrame = cFrame
#     prevMasked = masked
#
cv2.destroyAllWindows()
cap.release()

# import cv2
# import numpy as np
# import os
#
# videos = ['./edited/1.mp4', './edited/Jur_OK_1.mp4', './edited/kostka2.mp4']
# cap = cv2.VideoCapture(videos[0])
#
#
# if not cap.isOpened():
#     print("cannot read video input")
#     exit()
#
# # video metadata
# _, bgnd = cap.read()
# backGround, _, grayBackground = alterImage(bgnd)
# _, pFrame = cap.read()
# cFrame, prev_hsv, prev_gray = alterImage(pFrame)
# pBlur = cv2.GaussianBlur(prev_gray, (5, 5), 0)
# diff = cv2.absdiff(pBlur, pBlur)
# while True:
#     ret, cFrame = cap.read()
#     if not ret:
#         break
#     cFrame, _, gray = alterImage(cFrame)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     diff = cv2.absdiff(blur, pBlur)
#     _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     sumContours = 0
#     for contour in contours:
#         cntArea = cv2.contourArea(contour)
#         if cntArea > 300:
#             print("chyba streamu")
#         sumContours += cntArea
#     print(sumContours)
#     cv2.drawContours(cFrame, contours, -1, (0, 0, 255), 2)
#     cv2.imshow("frame", cFrame)
#     cv2.imshow("dil", thresh)
#     cv2.waitKey(50)
#     pFrame, prev_gray, pBlur = cFrame, gray, blur
#
# cv2.destroyAllWindows()
# cap.release()
