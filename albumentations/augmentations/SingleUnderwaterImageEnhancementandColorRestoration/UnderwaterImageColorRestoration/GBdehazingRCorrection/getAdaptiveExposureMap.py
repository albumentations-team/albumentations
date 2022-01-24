import numpy as np
import cv2

from GuidedFilter import GuidedFilter
from guidedfilter_He import guided_filter_he

np.seterr(over='ignore')
np.seterr(invalid ='ignore')
np.seterr(all ='ignore')


#
#  def AdaptiveExposureMap(img,sceneRadiance,Lambda,blockSize):
#     img = np.float32(img)
#     sceneRadiance = np.float32(sceneRadiance)
#
#     x = sceneRadiance * img + Lambda * (img **2 )
#     y = sceneRadiance ** 2 + Lambda *  (img ** 2)
#     S_x  = x / y
#
#     return S_x



def AdaptiveExposureMap(img, sceneRadiance, Lambda, blockSize):

    minValue = 10 ** -2
    img = np.uint8(img)
    sceneRadiance = np.uint8(sceneRadiance)

    YjCrCb = cv2.cvtColor(sceneRadiance, cv2.COLOR_BGR2YCrCb)
    YiCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    normYjCrCb = (YjCrCb - YjCrCb.min()) / (YjCrCb.max() - YjCrCb.min())
    normYiCrCb = (YiCrCb - YiCrCb.min()) / (YiCrCb.max() - YiCrCb.min())
    Yi = normYiCrCb[:, :, 0]
    Yj = normYjCrCb[:, :, 0]
    Yi = np.clip(Yi, minValue,1)
    Yj = np.clip(Yj, minValue,1)

    # print('np.min(Yi)',np.min(Yi))
    # print('np.max(Yi)',np.max(Yi))
    # print('np.min(Yj)',np.min(Yj))
    # print('np.max(Yj)',np.max(Yj))
    # Yi = YiCrCb[:, :, 0]
    # Yj = YjCrCb[:, :, 0]
    S = (Yj * Yi + 0.3 * Yi ** 2) / (Yj ** 2 + 0.3 * Yi ** 2)

    # print('S',S)

    gimfiltR = 50  # 引导滤波时半径的大小
    eps = 10 ** -3  # 引导滤波时epsilon的值

    # refinedS = guided_filter_he(YiCrCb, S, gimfiltR, eps)

    guided_filter = GuidedFilter(YiCrCb, gimfiltR, eps)
    # guided_filter = GuidedFilter(normYiCrCb, gimfiltR, eps)

    refinedS = guided_filter.filter(S)

    # print('guided_filter_he(YiCrCb, S, gimfiltR, eps)', refinedS)
    # S = np.clip(S, 0, 1)

    # cv2.imwrite('OutputImages_D/' + 'SSSSS' + '_GBdehazingRCorrectionStretching.jpg', np.uint8(S * 255))

    S_three = np.zeros(img.shape)
    S_three[:, :, 0] = S_three[:, :, 1] = S_three[:, :, 2] = refinedS

    return S_three