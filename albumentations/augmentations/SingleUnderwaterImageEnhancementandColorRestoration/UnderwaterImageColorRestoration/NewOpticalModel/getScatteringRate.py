import numpy as np

def getMinChannel(img,AtomsphericLight):
    img = np.float16(img)
    imgGrayNormalization = np.zeros((img.shape[0], img.shape[1]))
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            localMin = 1
            for k in range(0, 2):
                imgNormalization = img.item((i, j, k))/AtomsphericLight[k]
                if imgNormalization < localMin:
                    localMin = imgNormalization
            imgGrayNormalization[i, j] = localMin
    return imgGrayNormalization

def ScatteringRateMap(img,AtomsphericLight,blockSize):
    img = getMinChannel(img, AtomsphericLight)
    addSize = int((blockSize - 1) / 2)
    newHeight = img.shape[0] + blockSize - 1
    newWidth =  img.shape[1] + blockSize - 1
    # 中间结果
    imgMiddle = np.zeros((newHeight, newWidth))
    imgMiddle[:, :] = 1
    imgMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = img
    # print('imgMiddle',imgMiddle)
    imgDark = np.zeros((img.shape[0], img.shape[1]))
    for i in range(addSize, newHeight - addSize):
        for j in range(addSize, newWidth - addSize):
            localMin = 1
            for k in range(i - addSize, i + addSize + 1):
                for l in range(j - addSize, j + addSize + 1):
                    if imgMiddle.item((k, l)) < localMin:
                        localMin = imgMiddle.item((k, l))
            imgDark[i - addSize, j - addSize] = localMin
    return imgDark


# def getMinChannel(img):
#     img = np.float16(img)
#     imgGrayNormalization = np.zeros((img.shape[0], img.shape[1]))
#     for i in range(0, img.shape[0]):
#         for j in range(0, img.shape[1]):
#             localMin = 10
#             for k in range(0, 2):
#                 imgNormalization = img.item((i, j, k))
#                 if imgNormalization < localMin:
#                     localMin = imgNormalization
#             imgGrayNormalization[i, j] = localMin
#     return imgGrayNormalization
#
# def getDarkChannel(img,blockSize):
#     addSize = int((blockSize - 1) / 2)
#     newHeight = img.shape[0] + blockSize - 1
#     newWidth = img.shape[1] + blockSize - 1
#     imgMiddle = np.zeros((newHeight, newWidth))
#     imgMiddle[:, :] = 255
#     imgMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = img
#     # print('imgMiddle',imgMiddle)
#     imgDark = np.zeros((img.shape[0], img.shape[1]), np.float64)
#     for i in range(addSize, newHeight - addSize):
#         for j in range(addSize, newWidth - addSize):
#             localMin = 255
#             for k in range(i - addSize, i + addSize + 1):
#                 for l in range(j - addSize, j + addSize + 1):
#                     if imgMiddle.item((k, l)) < localMin:
#                         localMin = imgMiddle.item((k, l))
#             imgDark[i - addSize, j - addSize] = localMin
#     return imgDark
#
#
# def ScatteringRateMap(img,AtomsphericLight,blockSize):
#     img = np.float16(img)
#     img = img/AtomsphericLight
#     img[:, :, 0] = getDarkChannel(img[:,:,0],blockSize)
#     img[:, :, 1] = getDarkChannel(img[:,:,1],blockSize)
#     # print('img',img)
#     imgDark = getMinChannel(img)
#
#
#     return imgDark
