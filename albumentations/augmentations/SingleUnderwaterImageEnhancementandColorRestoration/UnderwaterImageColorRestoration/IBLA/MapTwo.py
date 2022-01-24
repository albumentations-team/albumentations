import numpy as np
from getOneChannelMax import getMaxChannel

def getGBMAxChannel(img):
    imgGray = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            localMax = 0
            for k in range(0, 2):
                if img.item((i, j, k)) > localMax:
                    localMax = img.item((i, j, k))
            imgGray[i, j] = localMax
    return imgGray


def R_minus_GB(img,blockSize,R_map):
    img = getGBMAxChannel(img)
    mip_map =  R_map - getMaxChannel(img, blockSize)
    return mip_map
