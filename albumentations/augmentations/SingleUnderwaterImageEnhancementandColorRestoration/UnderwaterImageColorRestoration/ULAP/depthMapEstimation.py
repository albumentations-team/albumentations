
import numpy as np

def depthMap(img):

    theta_0 = 0.51157954
    theta_1 = 0.50516165
    theta_2 = -0.90511117
    img = img / 255.0
    x_1 = np.maximum(img[:, :, 0], img[:, :, 1])
    x_2 = img[:, :, 2]
    Deptmap = theta_0 + theta_1 * x_1 + theta_2 * x_2

    return Deptmap



