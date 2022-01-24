
import numpy as np
import math
e = math.e


def getTransmission(d_f):

    transmission = e ** ( (-1/7)* d_f)

    transmission = np.clip(transmission, 0.1, 1)

    return transmission