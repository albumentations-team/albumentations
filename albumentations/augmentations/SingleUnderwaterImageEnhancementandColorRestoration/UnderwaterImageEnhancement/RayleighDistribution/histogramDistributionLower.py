import numpy as np
import math
import cv2


def histogram_rgbLower(RGB_array,height, width):
    RGB_min = np.min(RGB_array)
    RGB_max = np.max(RGB_array)
    # print('RGB_min',RGB_min)
    # print('RGB_max',RGB_max)
    RGB__middle = (RGB_max + RGB_min) / 2
    # print('RGB__middle', RGB__middle)
    array_upper_histogram_stretching = np.zeros((height, width))
    R_predicted_min = 255 * 0.05
    # print('R_predicted_min',R_predicted_min)

    for i in range(0, height):
        for j in range(0, width):
            if RGB_array[i][j] < RGB__middle:
                p_out = int((( RGB_array[i][j] - RGB_min) * ((255 - R_predicted_min) / (RGB_max - RGB__middle)) + R_predicted_min))
                array_upper_histogram_stretching[i][j] = p_out
            else:
                array_upper_histogram_stretching[i][j] = 255
    return array_upper_histogram_stretching

def histogramStretching_Lower(sceneRadiance, height, width):
    sceneRadiance = np.float64(sceneRadiance)
    b, g, r = cv2.split(sceneRadiance)
    R_array_Lower_histogram_stretching = histogram_rgbLower(r, height, width)
    G_array_Lower_histogram_stretching = histogram_rgbLower(g, height, width)
    B_array_Lower_histogram_stretching = histogram_rgbLower(b, height, width)
    # print('np.max(R_array_Lower_histogram_stretching)',np.max(R_array_Lower_histogram_stretching))
    # print('np.max(G_array_Lower_histogram_stretching)',np.max(G_array_Lower_histogram_stretching))
    # print('np.max(B_array_Lower_histogram_stretching)',np.max(B_array_Lower_histogram_stretching))

    sceneRadiance_Lower = np.zeros((height, width, 3))
    sceneRadiance_Lower[:, :, 0] = B_array_Lower_histogram_stretching
    sceneRadiance_Lower[:, :, 1] = G_array_Lower_histogram_stretching
    sceneRadiance_Lower[:, :, 2] = R_array_Lower_histogram_stretching
    sceneRadiance_Lower = np.uint8(sceneRadiance_Lower)

    return sceneRadiance_Lower

