import os

import datetime
import numpy as np
import cv2
import natsort

from DetermineDepth import determineDepth
from TransmissionEstimation import getTransmission
from getAdaptiveExposureMap import AdaptiveExposureMap
from getAdaptiveSceneRadiance import AdaptiveSceneRadiance
from getAtomsphericLight import getAtomsphericLight
from refinedTransmission import Refinedtransmission
from sceneRadianceGb import sceneRadianceGB
from sceneRadianceR import sceneradiance

np.seterr(over='ignore')
if __name__ == '__main__':
    pass

starttime = datetime.datetime.now()

folder = "C:/Users/Administrator/Desktop/UnderwaterImageEnhancement/Physical/GBdehazingRCorrection"
path = folder + "/InputImages"
files = os.listdir(path)
files = natsort.natsorted(files)

for i in range(len(files)):
    file = files[i]
    filepath = path + "/" + file
    prefix = file.split('.')[0]
    if os.path.isfile(filepath):
        print('********    file   ********', file)
        img = cv2.imread('InputImages/' + file)
        blockSize = 9

        YiCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        print('YiCrCb[:, :, 2]',YiCrCb[:, :, 2])
        normYiCrCb = (YiCrCb - YiCrCb.min()) / (YiCrCb.max() - YiCrCb.min())
        Yi = normYiCrCb[:, :, 0]
        Cr = normYiCrCb[:, :, 1]
        Cb = normYiCrCb[:, :, 2]


    cv2.imwrite('OutputImages/' + prefix + 'GBDehazedRcoorectionUDCPAdaptiveMap.jpg', np.uint8(np.clip((Yi*255), 0, 255)))
    cv2.imwrite('OutputImages/' + prefix + 'GBDehazedRcoorectionUDCPAdaptiveMapCr.jpg', np.uint8(np.clip((Cr*255), 0, 255)))
    cv2.imwrite('OutputImages/' + prefix + 'GBDehazedRcoorectionUDCPAdaptiveMapCb.jpg', np.uint8(np.clip((Cb*255), 0, 255)))

Endtime = datetime.datetime.now()
Time = Endtime - starttime
print('Time', Time)


