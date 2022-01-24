from .GuidedFilter import GuidedFilter

import os
import math
import numpy as np
import cv2


class Node(object):
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value

    def printInfo(self):
        print(self.x, self.y, self.value)


def getMinChannel(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        pass
    else:
        print("bad image shape, input must be color image")
        return None
    imgGray = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    localMin = 255

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            localMin = 255
            for k in range(0, 3):
                if img.item((i, j, k)) < localMin:
                    localMin = img.item((i, j, k))
            imgGray[i, j] = localMin
    return imgGray


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


def getDarkChannel(img, blockSize):
    # 输入检查
    if len(img.shape) == 2:
        pass
    else:
        print("bad image shape, input image must be two demensions")
        return None

    # blockSize检查
    if blockSize % 2 == 0 or blockSize < 3:
        print('blockSize is not odd or too small')
        return None
    # print('blockSize', blockSize)
    # 计算addSize
    addSize = int((blockSize - 1) / 2)
    newHeight = img.shape[0] + blockSize - 1
    newWidth = img.shape[1] + blockSize - 1

    # 中间结果
    imgMiddle = np.zeros((newHeight, newWidth))
    imgMiddle[:, :] = 255
    # print('imgMiddle',imgMiddle)
    # print('type(newHeight)',type(newHeight))
    # print('type(addSize)',type(addSize))
    imgMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = img
    # print('imgMiddle', imgMiddle)
    imgDark = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    localMin = 255

    for i in range(addSize, newHeight - addSize):
        for j in range(addSize, newWidth - addSize):
            localMin = 255
            for k in range(i - addSize, i + addSize + 1):
                for l in range(j - addSize, j + addSize + 1):
                    if imgMiddle.item((k, l)) < localMin:
                        localMin = imgMiddle.item((k, l))
            imgDark[i - addSize, j - addSize] = localMin

    return imgDark


def getAtomsphericLight(darkChannel, img, meanMode=False, percent=0.001):
    size = darkChannel.shape[0] * darkChannel.shape[1]
    height = darkChannel.shape[0]
    width = darkChannel.shape[1]

    nodes = []

    # 用一个链表结构(list)存储数据
    for i in range(0, height):
        for j in range(0, width):
            oneNode = Node(i, j, darkChannel[i, j])
            nodes.append(oneNode)

    # 排序
    nodes = sorted(nodes, key=lambda node: node.value, reverse=True)

    atomsphericLight = 0

    # 原图像像素过少时，只考虑第一个像素点
    if int(percent * size) == 0:
        for i in range(0, 3):
            if img[nodes[0].x, nodes[0].y, i] > atomsphericLight:
                atomsphericLight = img[nodes[0].x, nodes[0].y, i]
        return atomsphericLight

    # 开启均值模式
    if meanMode:
        sum = 0
        for i in range(0, int(percent * size)):
            for j in range(0, 3):
                sum = sum + img[nodes[i].x, nodes[i].y, j]

        atomsphericLight = int(sum / (int(percent * size) * 3))
        return atomsphericLight

    # 获取暗通道前0.1%(percent)的位置的像素点在原图像中的最高亮度值
    for i in range(0, int(percent * size)):
        for j in range(0, 3):
            if img[nodes[i].x, nodes[i].y, j] > atomsphericLight:
                atomsphericLight = img[nodes[i].x, nodes[i].y, j]

    return atomsphericLight


def getRecoverScene(img, omega=0.95, t0=0.1, blockSize=15, meanMode=False, percent=0.001):
    
    gimfiltR = 50  # 引导滤波时半径的大小
    eps = 10 ** -3  # 引导滤波时epsilon的值
    
    
    imgGray = getMinChannel(img)
    # print('imgGray', imgGray)
    imgDark = getDarkChannel(imgGray, blockSize=blockSize)
    atomsphericLight = getAtomsphericLight(imgDark, img, meanMode=meanMode, percent=percent)

    imgDark = np.float64(imgDark)
    transmission = 1 - omega * imgDark / atomsphericLight

    guided_filter = GuidedFilter(img, gimfiltR, eps)
    transmission = guided_filter.filter(transmission)
    
    # 防止出现t小于0的情况
    # 对t限制最小值为0.1

    transmission = np.clip(transmission, t0, 0.9)

    sceneRadiance = np.zeros(img.shape)
    for i in range(0, 3):
        img = np.float64(img)
        sceneRadiance[:, :, i] = (img[:, :, i] - atomsphericLight) / transmission + atomsphericLight

    sceneRadiance = np.clip(sceneRadiance, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)

    return transmission,sceneRadiance


# def dcp(img, isTM=False):
#     transmission, sceneRadiance = getRecoverScene(img)
#     if isTM:
#         return np.uint8(transmission * 255)
#     else:
#         return sceneRadiance


#############################################################

def getRGB_Darkchannel(img, blockSize):
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
    imgDark = np.float16(imgDark)

    return imgDark


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


def RecoverHE(sceneRadiance):
    for i in range(3):
        sceneRadiance[:, :, i] =  cv2.equalizeHist(sceneRadiance[:, :, i])

    return sceneRadiance


def sceneRadianceRGB(img, transmissionB, transmissionG, transmissionR, AtomsphericLight):
    transmission = np.zeros(img.shape)
    transmission[:, :, 0] = transmissionB
    transmission[:, :, 1] = transmissionG
    transmission[:, :, 2] = transmissionR
    sceneRadiance = np.zeros(img.shape)
    img = np.float32(img)
    for i in range(0, 3):
        sceneRadiance[:, :, i] = (img[:, :, i] - AtomsphericLight[i]) / transmission[:, :, i]  + AtomsphericLight[i]
        # 限制透射率 在0～255
    sceneRadiance = np.clip(sceneRadiance,0, 255)
    sceneRadiance = np.uint8(sceneRadiance)
    return sceneRadiance


def getAtomsphericLightDCP_Bright(darkChannel, img,percent):
    # print('getAtomsphericLightDCP_MAX(darkChannel, img,percent):',darkChannel)
    size = darkChannel.shape[0] * darkChannel.shape[1]
    height = darkChannel.shape[0]
    width = darkChannel.shape[1]
    nodes = []
    img = np.float32(img)
    # 用一个链表结构(list)存储数据
    for i in range(0, height):
        for j in range(0, width):
            oneNode = Node(i, j, darkChannel[i, j])
            nodes.append(oneNode)
    # 排序
    nodes = sorted(nodes, key=lambda node: node.value, reverse=True)
    atomsphericLight = 0
    # 获取暗通道前0.1%(percent)的位置的像素点在原图像中的最高亮度值
    SumImg = np.zeros(3)
    for i in range(0, int(percent * size)):
        SumImg  =  SumImg  + img[nodes[i].x, nodes[i].y, :]
    AtomsphericLight  = SumImg/(int(percent * size))
    return AtomsphericLight


def Refinedtransmission(transmissionB,transmissionG,transmissionR_Stretched,img):


    gimfiltR = 50  # 引导滤波时半径的大小
    eps = 10 ** -3  # 引导滤波时epsilon的值

    guided_filter = GuidedFilter(img, gimfiltR, eps)
    transmissionR_Stretched = guided_filter.filter(transmissionR_Stretched)
    transmissionG = guided_filter.filter(transmissionG)
    transmissionB = guided_filter.filter(transmissionB)

    transmission = np.zeros(img.shape)
    transmission[:, :, 0] = transmissionB
    transmission[:, :, 1] = transmissionG
    transmission[:, :, 2] = transmissionR_Stretched
    transmission = np.clip(transmission,0.2, 0.9)
    return transmission[:, :, 0], transmission[:, :, 1],transmission[:, :, 2]


def getGBTransmissionESt(transmissionR, AtomsphericLight):
    transmissionB  = transmissionMap(transmissionR, AtomsphericLight[0],450 , AtomsphericLight[2], 620)
    transmissionG  = transmissionMap(transmissionR, AtomsphericLight[1],540 , AtomsphericLight[2], 620)

    return transmissionB,transmissionG


def getTransmission(d_f):

    transmission = math.e ** ( (-1/7)* d_f)

    transmission = np.clip(transmission, 0.1, 1)

    return transmission


def closePoint(img, AtomsphericLight):
    Max = []
    img = np.float32(img)
    for i in range(0,3):
        Max_Abs =  np.absolute(img[i] - AtomsphericLight[i])
        Max_I = np.max(Max_Abs)
        Max_B = np.max([AtomsphericLight[i],(255 -AtomsphericLight[i])])
        temp  = Max_I / Max_B
        Max.append(temp)
    K_b = 1  - np.max(Max)
    return K_b


def global_stretching(img_L):
    height = len(img_L)
    width = len(img_L[0])
    length = height * width
    R_rray = []
    for i in range(height):
        for j in range(width):
            R_rray.append(img_L[i][j])
    R_rray.sort()
    I_min = R_rray[int(length / 200)]
    I_max = R_rray[-int(length / 200)]
    array_Global_histogram_stretching_L = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            if img_L[i][j] < I_min:
                p_out = img_L[i][j]
                array_Global_histogram_stretching_L[i][j] = 0.05
            elif (img_L[i][j] > I_max):
                p_out = img_L[i][j]
                array_Global_histogram_stretching_L[i][j] = 0.95
            else:
                p_out = (img_L[i][j] - I_min) * ((0.95-0.05) / (I_max - I_min))+ 0.05
                array_Global_histogram_stretching_L[i][j] = p_out
    return (array_Global_histogram_stretching_L)


def S(a,sigma):
    FinalS = (1 + math.e ** (-32 * (a - sigma))) ** -1
    return FinalS


def Scene_depth(d_R,d_D,d_B,img,AtomsphericLight):
    avg_BL = np.mean(AtomsphericLight)
    avg_Ir = np.mean(img[:,:,2])

    Theta_a = S(avg_BL, 0.5*255)
    Theta_b = S(avg_Ir, 0.1*255)

    Depth_map =   Theta_b *  (Theta_a * d_D  +  (1  - Theta_a) *  d_R )  +   (1 - Theta_b) *   d_B
    return Depth_map


def StretchingFusion(map):
    map_max = np.max(map)
    map_min = np.min(map)

    # if(map_max < 2):
    #     map_max = 5
    finalmap  = (map - map_min) / (map_max - map_min)
    return finalmap


def blurrnessMap(img, blockSize, n):
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


def Selection_SameFour(I_gray_Q,height_begin,height_end,width_begin,width_end):
    height = height_end - height_begin
    width  = width_end  - width_begin
    Q = []

    for i in range(height_begin,height_end):
        for j in range(width_begin,width_end):
            Q.append(I_gray_Q[i,j])
    Q = np.array(Q).reshape((height,width))
    return Q


def quadTree(img,I_gray_Q):
    height = I_gray_Q.shape[0]
    width = I_gray_Q.shape[1]
    if(height/2 == 1):
        height = height- 1
    if (width / 2  == 1 ):
        width = width - 1
    half_height = int(height/2)
    half_width = int(width/2)
    Q1 = Selection_SameFour(I_gray_Q,0,half_height,0,half_width)
    Q2 = Selection_SameFour(I_gray_Q,0,half_height,half_width,(half_width*2))
    Q3 = Selection_SameFour(I_gray_Q,half_height,(half_height*2),0,half_width)
    Q4 = Selection_SameFour(I_gray_Q,half_height,(half_height*2),half_width,(half_width*2))
    Q1_var = np.var(Q1)
    Q2_var = np.var(Q2)
    Q3_var = np.var(Q3)
    Q4_var = np.var(Q4)
    Q_var_min = np.min([Q1_var,Q2_var,Q3_var,Q4_var])
    if(Q1_var == Q_var_min):
        return img[0:half_height,0:half_width,:],Q1
    if (Q2_var == Q_var_min):
        return img[0:half_height,half_width:(half_width*2),:],Q2
    if (Q3_var == Q_var_min):
        return img[half_height:(half_height*2),0:half_width,:],Q3
    if (Q4_var == Q_var_min):
        return img[half_height:(half_height*2),half_width:(half_width*2),:],Q4


def getAtomsphericLightLb(img,blockSize,n):
    BlurrnessMap  =  blurrnessMap(img, blockSize, n)
    for i in range(0, 5):
        img, BlurrnessMap = quadTree(img, BlurrnessMap)
    AtomsphericLight = np.zeros(3)
    for i in range(3):
        AtomsphericLight[i] = np.mean(img[:, :, i])
    return AtomsphericLight


def getAtomsphericLightLv(img):
    I_gray_Q = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(0,5):
        img,I_gray_Q  = quadTree(img,I_gray_Q)
    AtomsphericLight = np.zeros(3)
    for i in range(3):
        AtomsphericLight[i] = np.mean(img[:,:,i])
    return AtomsphericLight


def S(img,sigma):
    height = img.shape[0]
    width = img.shape[1]
    Filter_more_half = []
    for i in range(height):
        for j in range(width):
            if(img[i,j]>(0.5*255)):
                Filter_more_half.append(img[i,j])
    Length_more_half = len(Filter_more_half)
    a = Length_more_half/(height * width)
    FinalS = (1 + math.e ** (-32 * (a - sigma))) ** -1
    return FinalS


def ThreeAtomsphericLightFusion(AtomsphericLightOne,AtomsphericLightTwo,AtomsphericLightThree,img):
    FialAtomsphericLightFusion = np.zeros(3)
    for i in range(0,3):
        Max = np.max([AtomsphericLightOne[i],AtomsphericLightTwo[i],AtomsphericLightThree[i]])
        Min = np.min([AtomsphericLightOne[i],AtomsphericLightTwo[i],AtomsphericLightThree[i]])
        alpha = S(img[:,:,i],sigma = 0.2)
        # print('alpha',alpha)
        AtomsphericLightFusion = alpha * Max +  (1-alpha) * Min
        FialAtomsphericLightFusion[i]= AtomsphericLightFusion
    return FialAtomsphericLightFusion


def max_R(img,blockSize):
    img = img[:,:,2]
    R_map  = getMaxChannel(img, blockSize)
    return R_map


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


def ibla(img, isHe = False):
    blockSize = 9
    n = 5
    RGB_Darkchannel = getRGB_Darkchannel(img, blockSize)
    BlurrnessMap = blurrnessMap(img, blockSize, n)
    AtomsphericLightOne = getAtomsphericLightDCP_Bright(RGB_Darkchannel, img, percent=0.001)
    AtomsphericLightTwo = getAtomsphericLightLv(img)
    AtomsphericLightThree = getAtomsphericLightLb(img, blockSize, n)
    AtomsphericLight = ThreeAtomsphericLightFusion(AtomsphericLightOne, 
                        AtomsphericLightTwo, AtomsphericLightThree, img)

    R_map = max_R(img, blockSize)
    mip_map = R_minus_GB(img, blockSize, R_map)
    bluriness_map = BlurrnessMap

    d_R = 1 - StretchingFusion(R_map)
    d_D = 1 - StretchingFusion(mip_map)
    d_B = 1 - StretchingFusion(bluriness_map)

    d_n = Scene_depth(d_R, d_D, d_B, img, AtomsphericLight)
    d_n_stretching = global_stretching(d_n)
    d_0 = closePoint(img, AtomsphericLight)
    d_f = 8  * (d_n +  d_0)


    transmissionR = getTransmission(d_f)
    transmissionB, transmissionG = getGBTransmissionESt(transmissionR, AtomsphericLight)
    
    transmissionB, transmissionG, transmissionR = Refinedtransmission(
                                                    transmissionB, transmissionG, 
                                                    transmissionR, img)

    sceneRadiance = sceneRadianceRGB(img, transmissionB, 
                                    transmissionG, transmissionR, 
                                    AtomsphericLight)

    if isHe:
        return RecoverHE(sceneRadiance)
    else:
        return sceneRadiance
#########################################################################


