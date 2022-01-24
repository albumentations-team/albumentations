import cv2
import numpy as np


def getMaxDarkChannel(img, blockSize):
    addSize = int((blockSize - 1) / 2)
    newHeight = img.shape[0] + blockSize - 1
    newWidth = img.shape[1] + blockSize - 1
    # 中间结果
    imgMiddle = np.zeros((newHeight, newWidth))
    imgMiddle[:, :] = 0
    imgMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = img
    # print('imgMiddle',imgMiddle)
    imgDark = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for i in range(addSize, newHeight - addSize):
        for j in range(addSize, newWidth - addSize):
            localMin = 0
            for k in range(i - addSize, i + addSize + 1):
                for l in range(j - addSize, j + addSize + 1):
                    if imgMiddle.item((k, l)) > localMin:
                        localMin = imgMiddle.item((k, l))
            imgDark[i - addSize, j - addSize] = localMin
    imgDark = np.uint8(imgDark)
    return imgDark


def blurrnessMap(img,blockSize,n):
    B = np.zeros((img.shape))
    for i in range(1, n):

        r = 2 ** i * (n - 1) + 1
        # print('r', r)
        img = np.uint8(img)
        blur = cv2.GaussianBlur(img, (r, r), r)
        blur = np.float32(blur)
        img = np.float32(img)
        B = np.absolute((img - blur)) + B
    B_Map = B / (n - 1)
    B_Map = np.uint8(B_Map)

    B_Map_dark = cv2.cvtColor((B_Map), cv2.COLOR_BGR2GRAY)

    Roughdepthmap = getMaxDarkChannel(B_Map_dark, blockSize)
    Refinedepthmap = cv2.bilateralFilter(Roughdepthmap, 9, 75, 75)
    return Refinedepthmap