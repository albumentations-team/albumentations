import numpy as np


def getMinChannel(img):
	imgGray = np.zeros((img.shape[0],img.shape[1]),'float32')
	for i in range(0,img.shape[0]):
		for j in range(0,img.shape[1]):
			localMin = 255
			for k in range(0,2):
				if img.item((i,j,k)) < localMin:
					localMin = img.item((i,j,k))
			imgGray[i,j] = localMin
	return imgGray

def getDarkChannel(img, blockSize):
    img = getMinChannel(img)
    addSize = int((blockSize - 1) / 2)
    newHeight = img.shape[0] + blockSize - 1
    newWidth = img.shape[1] + blockSize - 1
    # 中间结果
    imgMiddle = np.zeros((newHeight, newWidth))
    imgMiddle[:, :] = 255
    imgMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = img
    # print('imgMiddle',imgMiddle)
    imgDark = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for i in range(addSize, newHeight - addSize):
        for j in range(addSize, newWidth - addSize):
            localMin = 255
            for k in range(i - addSize, i + addSize + 1):
                for l in range(j - addSize, j + addSize + 1):
                    if imgMiddle.item((k, l)) < localMin:
                        localMin = imgMiddle.item((k, l))
            imgDark[i - addSize, j - addSize] = localMin
    return imgDark