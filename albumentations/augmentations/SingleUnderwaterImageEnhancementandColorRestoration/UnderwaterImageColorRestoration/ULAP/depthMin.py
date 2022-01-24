import numpy as np
import math


def  minDepth(img, BL):
    img = img/255.0
    BL = BL/255.0
    Max = []
    img = np.float32(img)
    for i in range(0,3):
        Max_Abs =  np.absolute(img[i] - BL[i])
        Max_I = np.max(Max_Abs)
        Max_B = np.max([BL[i],(1 -BL[i])])
        temp  = Max_I / Max_B
        Max.append(temp)
    K_b = np.max(Max)
    min_depth = 1 - K_b

    return min_depth



