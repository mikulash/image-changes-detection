import cv2
import numpy as np


def quantizationOfIMG(image, k, max_iter):
    # https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html
    i = np.float32(image).reshape(-1, 3)
    condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 1.0)
    ret, label, center = cv2.kmeans(i, k, None, condition, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    final_img = center[label.flatten()]
    final_img = final_img.reshape(image.shape)
    return final_img
