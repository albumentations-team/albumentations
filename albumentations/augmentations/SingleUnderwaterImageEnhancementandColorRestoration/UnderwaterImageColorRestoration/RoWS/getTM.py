import numpy as np
def getMinChannel(img,AtomsphericLight):
    imgGrayNormalization = np.zeros((img.shape[0], img.shape[1]))
    img = img * AtomsphericLight
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            localMin = 1
            for k in range(0, 3):
                imgNormalization = img.item((i, j, k))
                if imgNormalization < localMin:
                    localMin = imgNormalization
            imgGrayNormalization[i, j] = localMin
    return imgGrayNormalization

def getTransmission(img,AtomsphericLight ,blockSize):
    img = img/255
    AtomsphericLight = AtomsphericLight/255
    print('np.mean(img * AtomsphericLight)',np.mean(img * AtomsphericLight))
    img = getMinChannel(img,AtomsphericLight)
    print('np.mean(img)',np.mean(img))
    addSize = int((blockSize - 1) / 2)
    newHeight = img.shape[0] + blockSize - 1
    newWidth = img.shape[1] + blockSize - 1
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
    transmission  = 1- imgDark
    transmission = np.clip(transmission, 0.1, 0.9)

    return transmission