import numpy as np
import math
import cv2
from skimage.color import rgb2hsv,hsv2rgb


e = np.e
esp = 2.2204e-16


# 用于排序时存储原来像素点位置的数据结构
class Node(object):
	def __init__(self,x,y,value):
		self.x = x
		self.y = y
		self.value = value
	def printInfo(self):
		print(self.x,self.y,self.value)

def rayleighStrUpper(nodes, height, width,lower_Position):
    allSize = height * width
    alpha = 0.5
    selectedRange = [0, 255]
    NumPixel = np.zeros(256)
    temp = np.zeros(256)
    for i in range(lower_Position, allSize):
            NumPixel[nodes[i].value] = NumPixel[nodes[i].value] + 1
    ProbPixel = NumPixel / (allSize-lower_Position)
    CumuPixel = np.cumsum(ProbPixel)
    # print('UpperCumuPixel', CumuPixel)
    valSpread = selectedRange[1] - selectedRange[0]
    hconst = 2 * alpha ** 2
    vmax = 1 - e ** (-1 / hconst)
    val = vmax * (CumuPixel)
    val = np.array(val)

    for i in range(256):
        if (val[i] >= 1):
            val[i] = val[i] - esp
    for i in range(256):
        temp[i] = np.sqrt(-hconst * math.log((1 - val[i]), e))
        normalization = temp[i] * valSpread
        if(normalization > 255):
            CumuPixel[i] = 250
        else:
            CumuPixel[i] = normalization
    # print('UpperCumuPixel',CumuPixel)
    for i in range(lower_Position, allSize):
        nodes[i].value = CumuPixel[nodes[i].value]
    return nodes



def uperLower(r, height, width):
    allSize = height * width
    R_max = np.max(r)
    R_min = np.min(r)
    R__middle = (R_max - R_min) / 2 + R_min
    # R__middle = np.mean(r)
    nodes = []
    for i in range(0, height):
        for j in range(0, width):
            oneNode = Node(i, j, r[i, j])
            nodes.append(oneNode)
    nodes = sorted(nodes, key=lambda node: node.value, reverse=False)
    # print('R__middle',R__middle)

    for i in range(240000):
        if (nodes[i].value > R__middle):
            # print('nodes[i].value',nodes[i].value)
            middle_Position = i
            break
    lower_Position = middle_Position
    # print('lower_Position',lower_Position)

    for i in range(240000):
        nodes[i].value = np.int(nodes[i].value)
    # print('nodes[0].value)', nodes[0].value)
    # print('nodes[lower_Position + 2].value)', nodes[lower_Position + 2].value)

    nodesUpper  = rayleighStrUpper(nodes, height, width,lower_Position)

    array_upper_histogram_stretching = np.zeros((height, width))

    # print('lower_Position', lower_Position)
    # print('allSize', allSize)
    for i in range(0, allSize):
        if(i > lower_Position):
            array_upper_histogram_stretching[nodes[i].x, nodes[i].y] = nodesUpper[i].value
        else:
            array_upper_histogram_stretching[nodes[i].x, nodes[i].y] = 5

    # print('np.mean(array_upper_histogram_stretching))',np.mean(array_upper_histogram_stretching))

    return array_upper_histogram_stretching

def rayleighStretching_Upper(sceneRadiance, height, width):
    img_hsv = rgb2hsv(sceneRadiance)
    h, s, v = cv2.split(img_hsv)

    s = s * 255
    v = v * 255


    v_array_upper_histogram_stretching = uperLower(v, height, width)/255
    s_array_upper_histogram_stretching = uperLower(s, height, width)/255
    h_array_upper_histogram_stretching = h
    # print('R_array_upper_histogram_stretching',R_array_upper_histogram_stretching)

    sceneRadiance_Upper = np.zeros((height, width, 3))
    sceneRadiance_Upper[:, :, 0] = h_array_upper_histogram_stretching
    sceneRadiance_Upper[:, :, 1] = s_array_upper_histogram_stretching
    sceneRadiance_Upper[:, :, 2] = v_array_upper_histogram_stretching
    img_rgb = hsv2rgb(sceneRadiance_Upper) * 255
    for i in range(0, 3):
        for j in range(0, img_rgb.shape[0]):
            for k in range(0, img_rgb.shape[1]):
                if img_rgb[j, k, i] > 250:
                    img_rgb[j, k, i] = 250
                if img_rgb[j, k, i] < 5:
                    img_rgb[j, k, i] = 5

    sceneRadiance_Upper = np.uint8(img_rgb)

    return sceneRadiance_Upper

