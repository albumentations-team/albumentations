import cv2
from skimage.color import rgb2hsv,hsv2rgb
import numpy as np
from skimage.color import rgb2lab, lab2rgb

from global_StretchingL import global_stretching
from global_stretching_ab import global_Stretching_ab


def  LABStretching(sceneRadiance):


    sceneRadiance = np.clip(sceneRadiance, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)
    height = len(sceneRadiance)
    width = len(sceneRadiance[0])
    img_lab = rgb2lab(sceneRadiance)
    L, a, b = cv2.split(img_lab)

    img_L_stretching = global_stretching(L, height, width)
    img_a_stretching = global_Stretching_ab(a, height, width)
    img_b_stretching = global_Stretching_ab(b, height, width)

    labArray = np.zeros((height, width, 3), 'float64')
    labArray[:, :, 0] = img_L_stretching
    labArray[:, :, 1] = img_a_stretching
    labArray[:, :, 2] = img_b_stretching
    img_rgb = lab2rgb(labArray) * 255



    return img_rgb
















    sceneRadiance = np.clip(sceneRadiance, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)
    height = len(sceneRadiance)
    width = len(sceneRadiance[0])
    img_hsv = rgb2hsv(sceneRadiance)
    img_hsv[:, :, 1] = global_stretching(img_hsv[:, :, 1], height, width)
    img_hsv[:, :, 2] = global_stretching(img_hsv[:, :, 2], height, width)
    img_rgb = hsv2rgb(img_hsv) * 255

    return img_rgb