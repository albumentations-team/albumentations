import cv2
import numpy as np

def GetMaxR(img,blockSize):
    addSize = int((blockSize - 1) / 2)
    newHeight = img.shape[0] + blockSize - 1
    newWidth = img.shape[1] + blockSize - 1
    # 中间结果
    imgMiddle = np.zeros((newHeight, newWidth))
    imgMiddle[:, :] = 0
    imgMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = img
    # print('imgMiddle',imgMiddle)
    imgDark = np.zeros((img.shape[0], img.shape[1]))
    for i in range(addSize, newHeight - addSize):
        for j in range(addSize, newWidth - addSize):
            localMin = 0
            for k in range(i - addSize, i + addSize + 1):
                for l in range(j - addSize, j + addSize + 1):
                    if imgMiddle.item((k, l)) > localMin:
                        localMin = imgMiddle.item((k, l))
            imgDark[i - addSize, j - addSize] = localMin
    return imgDark


def TransmissionR(transmissionGB,img,blockSize):
    img = np.float16(img)/255
    MaxRChannel = GetMaxR(img[:,:,2],blockSize)
    # print('np.average(MaxRChannel)',np.average(MaxRChannel))
    # print('np.average(transmissionGB)',np.average(transmissionGB))
    alpha  = np.average(transmissionGB)/ np.average(MaxRChannel)
    transmissionR = alpha * MaxRChannel
    # transmissionR = np.clip(transmissionR, 0.1, 0.9)
    transmission = transmissionR
    # print('np.average(transmission)',np.average(transmission))
    return transmission





