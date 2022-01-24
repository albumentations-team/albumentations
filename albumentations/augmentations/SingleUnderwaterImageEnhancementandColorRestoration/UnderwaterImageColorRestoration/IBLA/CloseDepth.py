import numpy as np
import math


def  closePoint(img, AtomsphericLight):
    Max = []
    img = np.float32(img)
    for i in range(0,3):
        Max_Abs =  np.absolute(img[i] - AtomsphericLight[i])
        Max_I = np.max(Max_Abs)
        Max_B = np.max([AtomsphericLight[i],(255 -AtomsphericLight[i])])
        temp  = Max_I / Max_B
        Max.append(temp)
    print('Max',Max)
    K_b = 1  - np.max(Max)
    return K_b



