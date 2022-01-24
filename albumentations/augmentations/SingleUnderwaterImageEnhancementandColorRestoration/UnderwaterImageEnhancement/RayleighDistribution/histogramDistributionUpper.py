import numpy as np
import math
import cv2



def histogram_rgbUpper(RGB_array,height, width):
    RGB_min = np.min(RGB_array)
    RGB_max = np.max(RGB_array)
    RGB__middle = (RGB_max + RGB_min) / 2
    array_upper_histogram_stretching = np.zeros((height, width))
    R_predicted_max = 255 * 0.95

    for i in range(0, height):
        for j in range(0, width):
            if RGB_array[i][j] < RGB__middle:
                array_upper_histogram_stretching[i][j] = 0
            else:
                p_out = int((RGB_array[i][j] - RGB__middle) * ((R_predicted_max) / (RGB_max - RGB__middle)))
                array_upper_histogram_stretching[i][j] = p_out
    return array_upper_histogram_stretching

def histogramStretching_Upper(sceneRadiance, height, width):
    sceneRadiance = np.float64(sceneRadiance)
    b, g, r = cv2.split(sceneRadiance)
    R_array_upper_histogram_stretching = histogram_rgbUpper(r, height, width)
    G_array_upper_histogram_stretching = histogram_rgbUpper(g, height, width)
    B_array_upper_histogram_stretching = histogram_rgbUpper(b, height, width)
    # print('R_array_upper_histogram_stretching',R_array_upper_histogram_stretching)

    sceneRadiance_Upper = np.zeros((height, width, 3))
    sceneRadiance_Upper[:, :, 0] = B_array_upper_histogram_stretching
    sceneRadiance_Upper[:, :, 1] = G_array_upper_histogram_stretching
    sceneRadiance_Upper[:, :, 2] = R_array_upper_histogram_stretching
    sceneRadiance_Upper = np.uint8(sceneRadiance_Upper)

    return sceneRadiance_Upper

