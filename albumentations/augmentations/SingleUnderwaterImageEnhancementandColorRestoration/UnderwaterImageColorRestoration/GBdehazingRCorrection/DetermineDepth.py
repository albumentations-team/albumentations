import numpy as np

def getMAxChannel(img):
    imgGray = np.zeros((img.shape[0], img.shape[1]), 'float32')
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            localMax = 0
            for k in range(0, 2):
                if img.item((i, j, k)) > localMax:
                    localMax = img.item((i, j, k))
            imgGray[i, j] = localMax
    return imgGray


def getDarkChannel(img, blockSize):
    addSize = int((blockSize - 1) / 2)
    newHeight = img.shape[0] + blockSize - 1
    newWidth = img.shape[1] + blockSize - 1

    # 中间结果
    imgMiddle = np.zeros((newHeight, newWidth))
    imgMiddle[:, :] = 0
    imgMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = img
    # print('imgMiddle',imgMiddle)
    imgDark = np.zeros((img.shape[0], img.shape[1]), dtype=np.float16)
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


def determineDepth(img, blockSize):
    img2 = img/255
    img_GB = getMAxChannel(img2)
    Max_GB = getDarkChannel(img_GB, blockSize)
    Max_R  = getDarkChannel(img2[:,:,2], blockSize)
    largestDiff = Max_R  - Max_GB

    return largestDiff