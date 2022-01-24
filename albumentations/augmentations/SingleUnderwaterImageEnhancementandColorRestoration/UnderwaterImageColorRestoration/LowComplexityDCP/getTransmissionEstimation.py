import numpy as np
import cv2

def getTransmissionMap(img,AtomsphericLight ,blockSize):

    AtomsphericLight = np.array(AtomsphericLight)
    img = np.float64(img)
    imgDark = np.zeros((img.shape[0], img.shape[1]))
    imgGrayNormalization = np.zeros(img.shape)
    localMin = 1
    for k in range(0, 3):
        imgGrayNormalization[:, :, k] = img[:, :, k] / AtomsphericLight[k]
    imgUint8 = np.uint8((imgGrayNormalization[:, :, k] / np.max(imgGrayNormalization[:, :, k])) * 255)
    imgGrayNormalization[:, :, k] = np.float32(cv2.medianBlur(imgUint8, blockSize))/255

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            localMin = 1
            for k in range(0, 3):
                imgNormalization = imgGrayNormalization.item((i, j, k))
                if imgGrayNormalization.item((i, j, k)) < localMin:
                    localMin = imgNormalization
                    imgDark[i, j] = localMin

    transmission  = 1- imgDark
    transmission = np.clip(transmission, 0.1, 0.9)
    return transmission