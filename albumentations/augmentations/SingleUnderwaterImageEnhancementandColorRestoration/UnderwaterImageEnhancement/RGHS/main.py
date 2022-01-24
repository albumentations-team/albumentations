
# encoding=utf-8
import os
import numpy as np
import cv2
import natsort

from LabStretching import LABStretching
from color_equalisation import RGB_equalisation
from global_stretching_RGB import stretching
from relativeglobalhistogramstretching import RelativeGHstretching

np.seterr(over='ignore')
if __name__ == '__main__':
    pass


# folder = "C:/Users/Administrator/Desktop/UnderwaterImageEnhancement/NonPhysical/RGHS"
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
        # img = cv2.imread('InputImages/' + file)
        # path = np.unicode(path, 'utf-8')
        # img = cv2.imread('InputImages/' + file)
        # img = cv2.imread(np.unicode('InputImages/' + file, 'utf-8'))

        # print('img',img)

        height = len(img)
        width = len(img[0])
        # sceneRadiance = RGB_equalisation(img)

        sceneRadiance = img
        # sceneRadiance = RelativeGHstretching(sceneRadiance, height, width)

        sceneRadiance = stretching(sceneRadiance)


        sceneRadiance = LABStretching(sceneRadiance)


        cv2.imwrite('OutputImages/' + prefix + '_RGHS.jpg', sceneRadiance)
