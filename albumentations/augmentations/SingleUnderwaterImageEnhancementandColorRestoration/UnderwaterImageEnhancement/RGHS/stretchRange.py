from scipy import stats
import numpy as np


def stretchrange(r_array, height, width):

    length = height * width
    R_rray = r_array.flatten()
    R_rray.sort()
    print('R_rray', R_rray)
    mode = stats.mode(R_rray).mode[0]
    mode_index_before = list(R_rray).index(mode)
    # count = stats.mode(R_rray).count[0]

    SR_min = R_rray[int(mode_index_before * 0.005)]
    SR_max = R_rray[int(-(length - mode_index_before) * 0.005)]

    print('mode',mode)
    print('SR_min',SR_min)
    print('SR_max',SR_max)

    return SR_min, SR_max, mode