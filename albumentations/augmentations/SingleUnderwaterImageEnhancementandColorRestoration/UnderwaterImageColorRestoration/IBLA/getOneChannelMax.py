import numpy as np


def getMaxChannel(img, blockSize):
    addSize = int((blockSize - 1) / 2)
    newHeight = img.shape[0] + blockSize - 1
    newWidth = img.shape[1] + blockSize - 1

    # 中间结果
    imgMiddle = np.zeros((newHeight, newWidth),'float64')
    imgMiddle[:, :] = 0
    imgMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = img
    # print('imgMiddle',imgMiddle)
    imgDark = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)
    localMax = 0
    for i in range(addSize, newHeight - addSize):
        for j in range(addSize, newWidth - addSize):
            localMax = 0
            for k in range(i - addSize, i + addSize + 1):
                for l in range(j - addSize, j + addSize + 1):
                    if imgMiddle.item((k, l)) > localMax:
                        localMax = imgMiddle.item((k, l))
            imgDark[i - addSize, j - addSize] = localMax
    return imgDark