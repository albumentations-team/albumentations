import os
import numpy as np
import cv2
import natsort
import matplotlib.pyplot as plt


from TransmissionMap import TransmissionComposition
from getAtomsphericLight import getAtomsphericLight
from getColorContrastEnhancement import ColorContrastEnhancement
from getRGBDarkChannel import getDarkChannel
from getSceneRadiance import SceneRadiance





######################## Based on the DCP and the 0.1% brightest point is incorrect ########################
######################## Based on the DCP and the 0.1% brightest point is incorrect ########################
######################## Based on the DCP and the 0.1% brightest point is incorrect  and further cause the distortion of the restored images ########################
from getTransmissionEstimation import getTransmissionMap

np.seterr(over='ignore')
if __name__ == '__main__':
    pass

# folder = "C:/Users/Administrator/Desktop/UnderwaterImageEnhancement/Physical/LowComplexityDCP"
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

        blockSize = 9

        imgGray = getDarkChannel(img, blockSize)
        AtomsphericLight = getAtomsphericLight(imgGray, img, meanMode=True, percent=0.001)
        # print('AtomsphericLight',AtomsphericLight)
        transmission = getTransmissionMap(img, AtomsphericLight, blockSize)
        sceneRadiance = SceneRadiance(img, AtomsphericLight, transmission)
        sceneRadiance = ColorContrastEnhancement(sceneRadiance)

        cv2.imwrite('OutputImages/' + prefix + '_LowComplexityDCPMap.jpg', np.uint8(transmission * 255))
        cv2.imwrite('OutputImages/' + prefix + '_LowComplexityDCP.jpg', sceneRadiance)

        #
        # plt.imshow(np.uint8(img))
        # plt.show()


