import numpy as np
import cv2


def RGB_equalisation(img,height,width):

    img = np.float32(img)
    b, g, r = cv2.split(img)
    r_avg = np.mean(r)
    g_avg = np.mean(g)
    b_avg = np.mean(b)

    All_avg = np.array((r_avg,g_avg,b_avg))
    All_max = np.max(All_avg)
    All_min = np.min(All_avg)
    All_median = np.median(All_avg)
    A = All_median/All_min
    B = All_median/All_max

    if (All_min == r_avg):
        r = r * A
    if (All_min == g_avg):
        g = g * A
    if (All_min == b_avg):
        b = b * A

    if (All_max == r_avg):
        r = r * B
    if (All_max == g_avg):
        g = g * B
    if (All_max == b_avg):
        b = b * B


    sceneRadiance = np.zeros((height, width, 3), 'float64')
    sceneRadiance[:, :, 0] = b
    sceneRadiance[:, :, 1] = g
    sceneRadiance[:, :, 2] = r
    sceneRadiance = np.clip(sceneRadiance, 0, 255)


    return sceneRadiance







# a_r = r_avg/128
# r_avg  = np.mean(r)
# r_rray = cal_equalisation(r_rray,a_r)
# R_rray = cal_equalisation(R_rray,a_r)


