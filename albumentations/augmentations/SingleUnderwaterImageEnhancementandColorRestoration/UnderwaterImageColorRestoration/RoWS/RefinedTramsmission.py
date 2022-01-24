import numpy as np
from GuidedFilter import GuidedFilter


def  Refinedtransmission(transmission,img):

    gimfiltR = 50  # 引导滤波时半径的大小
    eps = 10 ** -3  # 引导滤波时epsilon的值

    guided_filter = GuidedFilter(img, gimfiltR, eps)
    transmission = guided_filter.filter(transmission)
    transmission = np.clip(transmission,0.1, 0.9)

    return transmission
