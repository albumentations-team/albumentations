import numpy as np
import math
# def getMinChannel(img,AtomsphericLight):
#     img = np.float32(img)/AtomsphericLight
#     imgGrayNormalization = np.zeros((img.shape[0], img.shape[1]))
#     for i in range(0, img.shape[0]):
#         for j in range(0, img.shape[1]):
#             localMin = 1
#             for k in range(0, 2):
#                 imgNormalization = img.item((i, j, k))
#                 if imgNormalization < localMin:
#                     localMin = imgNormalization
#             imgGrayNormalization[i, j] = localMin
#     return imgGrayNormalization
#
#
# # def getMinChannel(img,AtomsphericLight):
# #     imgGrayNormalization = np.zeros((img.shape[0], img.shape[1]))
# #     for i in range(0, img.shape[0]):
# #         for j in range(0, img.shape[1]):
# #             localMin = 1
# #             for k in range(0, 2):
# #                 imgNormalization = img.item((i, j, k)) / AtomsphericLight[k]
# #                 if imgNormalization < localMin:
# #                     localMin = imgNormalization
# #             imgGrayNormalization[i, j] = localMin
# #     return imgGrayNormalization
#
#
#
#
# def getTransmission(img,AtomsphericLight ,blockSize):
#     img = getMinChannel(img,AtomsphericLight)
#     addSize = int((blockSize - 1) / 2)
#     newHeight = img.shape[0] + blockSize - 1
#     newWidth = img.shape[1] + blockSize - 1
#     # 中间结果
#     imgMiddle = np.zeros((newHeight, newWidth))
#     imgMiddle[:, :] = 1
#     imgMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = img
#     # print('imgMiddle',imgMiddle)
#     imgDark = np.zeros((img.shape[0], img.shape[1]))
#     for i in range(addSize, newHeight - addSize):
#         for j in range(addSize, newWidth - addSize):
#             localMin = 1
#             for k in range(i - addSize, i + addSize + 1):
#                 for l in range(j - addSize, j + addSize + 1):
#                     if imgMiddle.item((k, l)) < localMin:
#                         localMin = imgMiddle.item((k, l))
#             imgDark[i - addSize, j - addSize] = localMin
#     transmission  = 1- imgDark
#
#     transmission = np.clip(transmission, 0.1, 0.9)
#     return transmission




def getTransmission(normI,AtomsphericLight ,w):
    M, N, C = normI.shape #M are the rows, N are the columns, C is the bgr channel
    B = AtomsphericLight
    padwidth = math.floor(w/2)
    padded = np.pad(normI/B, ((padwidth, padwidth), (padwidth, padwidth),(0,0)), 'constant')
    transmission = np.zeros((M,N,2))
    for y, x in np.ndindex(M, N):
        transmission[y,x,0] = 1 - np.min(padded[y : y+w , x : x+w , 0])
        transmission[y,x,1] = 1 - np.min(padded[y : y+w , x : x+w , 1])
    return transmission