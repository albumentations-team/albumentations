import cv2
import numpy as np

from GuidedFilter import GuidedFilter


def  Refinedtransmission(transmissionB,transmissionG,transmissionR_Stretched,img):


    gimfiltR = 50  # 引导滤波时半径的大小
    eps = 10 ** -3  # 引导滤波时epsilon的值

    guided_filter = GuidedFilter(img, gimfiltR, eps)
    transmissionR_Stretched = guided_filter.filter(transmissionR_Stretched)
    transmissionG = guided_filter.filter(transmissionG)
    transmissionB = guided_filter.filter(transmissionB)

    transmission = np.zeros(img.shape)
    transmission[:, :, 0] = transmissionB
    transmission[:, :, 1] = transmissionG
    transmission[:, :, 2] = transmissionR_Stretched
    transmission = np.clip(transmission,0.2, 0.9)
    return transmission[:, :, 0], transmission[:, :, 1],transmission[:, :, 2]

    # transmissionB = FilterTran(transmissionB,0.1,0.9)
    # transmissionG = FilterTran(transmissionG,0.25,0.95)
    # transmissionR = FilterTran(transmissionR,0.35,0.975)
    # transmissionB = FilterTran(transmissionB, 0.2, 0.9, 15, 95)
    # transmissionG = FilterTran(transmissionG, 0.25, 0.95, 15, 95)
    # transmissionR = FilterTran(transmissionR, 0.35, 0.975, 15, 95)
    # print('transmissionB',transmissionB)