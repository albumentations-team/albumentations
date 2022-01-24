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
from refinedTransmission import refinedtransmission

from sceneRadianceGb import sceneRadianceGB
from sceneRadianceR import sceneradiance

# # # # # # # # # # # # # # # # # # # # # # Normalized implement is necessary part as the fore-processing   # # # # # # # # # # # # # # # #

np.seterr(over='ignore')
if __name__ == '__main__':
    pass

starttime = datetime.datetime.now()

# folder = "C:/Users/Administrator/Desktop/UnderwaterImageEnhancement/Physical/GBdehazingRCorrection"
folder = "C:/Users/Administrator/Desktop/Databases/Dataset"
path = folder + "/InputImages"
files = os.listdir(path)
files =  natsort.natsorted(files)

for i in range(len(files)):
    file = files[i]
    filepath = path + "/" + file
    prefix = file.split('.')[0]
    if os.path.isfile(filepath):
        print('********    file   ********',file)
        img = cv2.imread(folder +'/InputImages/' + file)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        blockSize = 9
        largestDiff = determineDepth(img, blockSize)
        AtomsphericLight, AtomsphericLightGB, AtomsphericLightRGB = getAtomsphericLight(largestDiff, img)
        print('AtomsphericLightRGB',AtomsphericLightRGB)
        
        # transmission = getTransmission(img, AtomsphericLightRGB, blockSize=blockSize)
        transmission = getTransmission(img, AtomsphericLightRGB, blockSize)
        # print('transmission.shape',transmission.shape)
        # TransmissionComposition(folder, transmission, number, param='coarse')
        transmission = refinedtransmission(transmission, img)

        cv2.imwrite('OutputImages/' + prefix + '_GBDehazedRcoorectionUDCP_TM.jpg', np.uint8(transmission[:, :, 0] * 255))


        # TransmissionComposition(folder, transmission, number, param='refined_15_175_175')
        sceneRadiance_GB = sceneRadianceGB(img, transmission, AtomsphericLightRGB)

        # cv2.imwrite('OutputImages/' + prefix + 'GBDehazed.jpg', sceneRadiance_GB)


        # # print('sceneRadiance_GB',sceneRadiance_GB)
        sceneRadiance = sceneradiance(img, sceneRadiance_GB)
        # sceneRadiance= sceneRadiance_GB
        # cv2.imwrite('OutputImages/'+ prefix + 'GBDehazedRcoorectionUDCP.jpg', sceneRadiance)
        # # print('np.min(sceneRadiance)',np.min(sceneRadiance))
        # # print('sceneRadiance',sceneRadiance)
        
        S_x = AdaptiveExposureMap(img, sceneRadiance, Lambda=0.3, blockSize=blockSize)
        # print('S_x',S_x)
        sceneRadiance = AdaptiveSceneRadiance(sceneRadiance, S_x)
        
        # print('sceneRadiance',sceneRadiance)
        cv2.imwrite('OutputImages/' + prefix + 'GBDehazedRcoorectionUDCPAdaptiveMap.jpg', sceneRadiance)


Endtime = datetime.datetime.now()
Time = Endtime - starttime
print('Time', Time)


