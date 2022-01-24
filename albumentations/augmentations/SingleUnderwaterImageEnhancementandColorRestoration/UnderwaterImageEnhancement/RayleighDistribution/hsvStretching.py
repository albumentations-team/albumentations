import cv2
from skimage.color import rgb2hsv,hsv2rgb
import numpy as np

from global_Stretching_SV import global_stretching


def  HSVStretching(sceneRadiance):
    sceneRadiance = np.clip(sceneRadiance, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)
    height = len(sceneRadiance)
    width = len(sceneRadiance[0])
    img_hsv = rgb2hsv(sceneRadiance)
    img_hsv[:, :, 1] = global_stretching(img_hsv[:, :, 1], height, width)
    img_hsv[:, :, 2] = global_stretching(img_hsv[:, :, 2], height, width)
    img_rgb = hsv2rgb(img_hsv) * 255

    return img_rgb