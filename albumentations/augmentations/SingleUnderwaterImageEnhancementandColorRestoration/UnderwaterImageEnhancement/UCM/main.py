import os
import numpy as np
import cv2
import natsort
import xlwt
import datetime

from color_equalisation import RGB_equalisation
from global_histogram_stretching import stretching
from hsvStretching import HSVStretching
from sceneRadiance import sceneRadianceRGB


np.seterr(over='ignore')
if __name__ == '__main__':
    pass

# folder = "C:/Users/Administrator/Desktop/UnderwaterImageEnhancement/NonPhysical/UCM"
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
        # img = cv2.imread('InputImages/' + file)
        img = cv2.imread(folder + '/InputImages/' + file)
        # print('Number',Number)
        sceneRadiance = RGB_equalisation(img)
        sceneRadiance = stretching(sceneRadiance)
        # # cv2.imwrite(folder + '/OutputImages/' + Number + 'Stretched.jpg', sceneRadiance)
        sceneRadiance = HSVStretching(sceneRadiance)
        sceneRadiance = sceneRadianceRGB(sceneRadiance)
        cv2.imwrite('OutputImages/' + prefix + '_UCM.jpg', sceneRadiance)

endtime = datetime.datetime.now()
time = endtime-starttime
print('time',time)
