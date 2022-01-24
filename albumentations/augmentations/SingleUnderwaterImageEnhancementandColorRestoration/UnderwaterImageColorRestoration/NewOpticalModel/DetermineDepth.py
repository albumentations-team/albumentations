import numpy as np

def getMAxChannel(B_Dark,G_Dark):
    imgGray = np.zeros((B_Dark.shape), dtype=np.float16)
    for i in range(0, B_Dark.shape[0]):
        for j in range(0, B_Dark.shape[1]):
            imgGray[i, j] = max(B_Dark[i, j], G_Dark[i, j])
    return imgGray



def getDarkChannel(img, blockSize):
    addSize = int((blockSize - 1) / 2)
    newHeight = img.shape[0] + blockSize - 1
    newWidth = img.shape[1] + blockSize - 1

    # 中间结果
    imgMiddle = np.zeros((newHeight, newWidth))
    imgMiddle[:, :] = 1
    imgMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = img
    # print('imgMiddle',imgMiddle)
    imgDark = np.zeros((img.shape[0], img.shape[1]), dtype=np.float16)
    localMax = 1
    for i in range(addSize, newHeight - addSize):
        for j in range(addSize, newWidth - addSize):
            localMax = 1
            for k in range(i - addSize, i + addSize + 1):
                for l in range(j - addSize, j + addSize + 1):
                    if imgMiddle.item((k, l)) < localMax:
                        localMax = imgMiddle.item((k, l))
            imgDark[i - addSize, j - addSize] = localMax
    return imgDark


def determineDepth(img, blockSize):
    img = np.float16(img)
    img = img/255

    R_Dark = getDarkChannel(img[:, :, 2], blockSize)
    G_Dark = getDarkChannel(img[:, :, 1], blockSize)
    B_Dark = getDarkChannel(img[:,:,0], blockSize)
    GB_Max = getMAxChannel(B_Dark,G_Dark)
    largestDiff = R_Dark  - GB_Max
    return largestDiff
