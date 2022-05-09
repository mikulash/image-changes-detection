# Author: Mikulas Heinz, xheinz, 2022, License: AGPLv3

import cv2
import numpy as np
import os
from statistics import mean

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub


def avgError(threshold, imageChanges):
    chyba = 0
    if len(imageChanges) > 5:
        if mean(imageChanges) > threshold:
            chyba = 1
            print(mean(imageChanges))
        elif mean(imageChanges) < 50:
            print(mean(imageChanges))
            chyba = -1
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



def alterImage(frame, proportions=(500, 400)):
    """returns original image, hsv variant and greyscale variant"""
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

# nepouzito
def getMaskFromContours(grayImage, position, clrd):
    clrd = clrd.copy()
    obr = grayImage.copy()
    sharpeningKernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(obr, -1, sharpeningKernel)
    kernel = np.ones((2, 2), np.uint8)
    bluredSharp = cv2.bilateralFilter(sharp, 9, 75, 75)
    edgesSharp = cv2.Canny(bluredSharp, 120, 255, 1)
    diluted = cv2.dilate(edgesSharp, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(diluted, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
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
    return np.uint8(template)


"""main program"""
# read stream
videos = ['./edited/2.mp4', './edited/Jur_OK_1.mp4', './edited/kostka2.mp4']  # examples of videos
tolerance_horni = 1000
tolerance_dolni = 10
cap = cv2.VideoCapture(videos[0])

if not cap.isOpened():
    print("cannot read video input")
    exit()

# video metadata
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("frame_width", frame_width)
print("frame_height", frame_height)

"""get color from image to detect
right arrow to next frame, esq to continue detection
"""
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
mask = cv2.inRange(hsv, lowColor, upperColor)
prevMasked = cv2.bitwise_and(frame, frame, mask=mask)
"""MAIN LOOP"""
pFrame = frame
imageChanges = []
while True:
    ret, cFrame = cap.read()
    if not ret:
        break
    #     vyhladit plochy
    cFrame = cv2.bilateralFilter(cFrame, 5, 75, 75)
    # prevest do hsv a grayscale
    cFrame, hsv, gray = alterImage(cFrame, finalWidth)
    # nechat jen pixely v ramci mezich hsv barvy
    hueMask = cv2.inRange(hsv, lowColor, upperColor)
    masked = cv2.bitwise_and(cFrame, cFrame, mask=hueMask)
    """Detect errors here!!!"""
    # rozdil dvou poslednich snimku
    diff = cv2.absdiff(prevMasked, masked)
    # ponechat jenom Value z Hue-Saturation-Valie
    h, s, grayFromHUE = cv2.split(diff)
    # eliminovat male zmeny napr. z duvodu pohybu kamery
    blur = cv2.GaussianBlur(grayFromHUE, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    # zvyraznit vetsi zmeny
    dilated = cv2.dilate(thresh, None, iterations=3)
    # spocitat velikost zmen
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sumContours = 0
    for contour in contours:
        sumContours += cv2.contourArea(contour)
        if cv2.contourArea(contour) < 900:
            continue
    print(sumContours)
    # zjistit zmeny na poslednich nekolika snimcich
    imageChanges.append(sumContours)
    chyba = avgError(tolerance_horni, imageChanges)
    if sumContours > tolerance_horni and chyba == 1:
        cv2.putText(cFrame, "Status: Error", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)
        cv2.drawContours(kopie, contours, -1, (0, 0, 255), 2)
    if sumContours < tolerance_dolni and chyba == -1:
        cv2.putText(cFrame, "Status: Run out of filament", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

    kopie = cFrame.copy()

    cv2.imshow("masked", masked)
    cv2.imshow("kopie", kopie)
    cv2.waitKey(50)
    pFrame = cFrame
    prevMasked = masked

cv2.destroyAllWindows()
cap.release()
