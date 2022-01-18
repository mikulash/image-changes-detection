import cv2
from matplotlib import pyplot as plt


def fixColor(image):
    return(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


cap = cv2.VideoCapture('./edited/2.mp4')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1280, 720))

ret, frame0 = cap.read()
greyStartFrame = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
ret, tresh = cv2.threshold(greyStartFrame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret, frame1 = cap.read()
ret, frame2 = cap.read()

frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
ret, tframe1 = cv2.threshold(frame1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
dframe1 = cv2.absdiff(tframe1, tresh)
dframe1 = cv2.bilateralFilter(dframe1, 5,75,75)
plt.imshow(fixColor(dframe1))

while cap.isOpened() and ret:
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    ret, tframe2 = cv2.threshold(frame2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    dframe2 = cv2.absdiff(tframe2, tresh)
    dframe2 = cv2.bilateralFilter(dframe2, 5,75,75)
    diff = cv2.absdiff(dframe1, dframe2)
    blur = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sumContours = 0
    for contour in contours:
        sumContours += cv2.contourArea(contour)
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 300:
            continue
        cv2.rectangle(dframe1, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(dframe1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (120, 120, 120), 3)
    print(sumContours)
    """
        if sumContours > 150000:
        plt.imshow(fixColor(frame2))
        plt.show()
        exit()
    """

    #cv2.drawContours(dframe1, contours, -1, (0, 255, 0), 2)

    image = cv2.resize(dframe1, (1280, 720))
    out.write(image)
    cv2.imshow("feed", dframe1)
    dframe1 = dframe2
    ret, frame2 = cap.read()
    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
out.release()
