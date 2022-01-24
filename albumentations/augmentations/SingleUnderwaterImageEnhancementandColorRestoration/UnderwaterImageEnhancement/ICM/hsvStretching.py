import cv2
from skimage.color import rgb2hsv,hsv2rgb
import numpy as np

from global_Stretching import global_stretching


def  HSVStretching(sceneRadiance):
    height = len(sceneRadiance)
    width = len(sceneRadiance[0])
    img_hsv = rgb2hsv(sceneRadiance)
    h, s, v = cv2.split(img_hsv)
    img_s_stretching = global_stretching(s, height, width)
    # print('np.min(img_s_stretching)',np.min(img_s_stretching))
    # print('np.max(img_s_stretching)',np.max(img_s_stretching))
    # print('np.mean(img_s_stretching)',np.mean(img_s_stretching))



    img_v_stretching = global_stretching(v, height, width)

    # print('np.min(img_v_stretching)', np.min(img_v_stretching))
    # print('np.max(img_v_stretching)', np.max(img_v_stretching))
    # print('np.mean(img_v_stretching)', np.mean(img_v_stretching))

    # img_L_sHretching = global_Stretching.global_stretching(L, height, width)
    # img_a_stretching = global_stretching_ab.global_Stretching_ab(a, height, width)
    # img_b_stretching = global_stretching_ab.global_Stretching_ab(b, height, width)

    labArray = np.zeros((height, width, 3), 'float64')
    labArray[:, :, 0] = h
    labArray[:, :, 1] = img_s_stretching
    labArray[:, :, 2] = img_v_stretching
    img_rgb = hsv2rgb(labArray) * 255

    # img_rgb = np.clip(img_rgb, 0, 255)

    return img_rgb