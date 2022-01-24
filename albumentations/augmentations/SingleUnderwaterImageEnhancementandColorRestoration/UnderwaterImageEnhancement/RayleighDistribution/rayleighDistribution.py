import numpy as np
import math

e = np.e
esp = 2.2204e-16



class NodeLower(object):
	def __init__(self,x,y,value):
		self.x = x
		self.y = y
		self.value = value
	def printInfo(self):
		print(self.x,self.y,self.value)



# 用于排序时存储原来像素点位置的数据结构
class Node(object):
	def __init__(self,x,y,value):
		self.x = x
		self.y = y
		self.value = value
	def printInfo(self):
		print(self.x,self.y,self.value)

def rayleighStrLower(nodes, height, width,lower_Position):
    alpha = 0.4
    selectedRange = [0, 255]
    NumPixel = np.zeros(256)
    temp = np.zeros(256)
    for i in range(0, lower_Position):
        # print('nodes[i].value',type(nodes[i].value))
        NumPixel[nodes[i].value] = NumPixel[nodes[i].value] + 1
    ProbPixel = NumPixel / lower_Position
    CumuPixel = np.cumsum(ProbPixel)
    # print('CumuPixel',CumuPixel)

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
            CumuPixel[i] = 255
        else:
            CumuPixel[i] = normalization
    for i in range(0, lower_Position):
        nodes[i].value = CumuPixel[nodes[i].value]
    return nodes


def rayleighStrUpper(nodes, height, width,lower_Position):
    allSize = height*width
    alpha = 0.4
    selectedRange = [0, 255]
    NumPixel = np.zeros(256)
    temp = np.zeros(256)
    for i in range(lower_Position, allSize):
            NumPixel[nodes[i].value] = NumPixel[nodes[i].value] + 1
    ProbPixel = NumPixel / (allSize-lower_Position)
    CumuPixel = np.cumsum(ProbPixel)
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
            CumuPixel[i] = 255
        else:
            CumuPixel[i] = normalization
    for i in range(lower_Position, allSize):
        nodes[i].value = CumuPixel[nodes[i].value]
    return nodes



def uperLower(r, height, width):
    allSize = height * width
    R_max = np.max(r)
    R_min = np.min(r)
    R__middle = (R_max - R_min) / 2 + R_min
    R__middle = np.mean(r)
    node_upper = []
    node_lower = []
    for i in range(0, height):
        for j in range(0, width):
            oneNode = Node(i, j, r[i, j])
            oneNodeLower = NodeLower(i, j, r[i, j])
            node_upper.append(oneNode)
            node_lower.append(oneNodeLower)
    node_upper = sorted(node_upper, key=lambda node: node.value, reverse=False)
    node_lower = sorted(node_lower, key=lambda node: node.value, reverse=False)

    # print('R__middle',R__middle)
    # middle_Position=[]
    for i in range(allSize):
        if (node_upper[i].value > R__middle):
            # print('nodes[i].value',nodes[i].value)
            middle_Position = i
            break
    lower_Position = middle_Position
    # print('lower_Position',lower_Position)

    for i in range(allSize):
        node_upper[i].value = np.int(node_upper[i].value)
        node_lower[i].value = np.int(node_lower[i].value)
    # print('nodes', nodes[0].value)
    # print('nodes[lower_Position + 10].value', nodes[lower_Position + 2].value)

    nodesLower  = rayleighStrLower(node_lower, height, width,lower_Position)
    nodesUpper  = rayleighStrUpper(node_upper, height, width,lower_Position)

    array_lower_histogram_stretching = np.zeros((height, width))
    array_upper_histogram_stretching = np.zeros((height, width))


    for i in range(0, allSize):
        if(i > lower_Position):
            array_upper_histogram_stretching[nodesUpper[i].x, nodesUpper[i].y] = nodesUpper[i].value
            array_lower_histogram_stretching[nodesUpper[i].x, nodesUpper[i].y] = 255
        else:
            array_lower_histogram_stretching[nodesLower[i].x, nodesLower[i].y] = nodesLower[i].value
            array_upper_histogram_stretching[nodesLower[i].x, nodesLower[i].y] = 0

    # print('np.mean(array_lower_histogram_stretching))',np.mean(array_lower_histogram_stretching))
    # print('np.mean(array_upper_histogram_stretching))',np.mean(array_upper_histogram_stretching))

    return array_lower_histogram_stretching,array_upper_histogram_stretching

def rayleighStretching(sceneRadiance, height, width):

    R_array_lower_histogram_stretching, R_array_upper_histogram_stretching = uperLower(sceneRadiance[:, :, 2], height, width)
    G_array_lower_histogram_stretching, G_array_upper_histogram_stretching = uperLower(sceneRadiance[:, :, 1], height, width)
    B_array_lower_histogram_stretching, B_array_upper_histogram_stretching = uperLower(sceneRadiance[:, :, 0], height, width)

    sceneRadiance_Lower = np.zeros((height, width, 3), )
    sceneRadiance_Lower[:, :, 0] = B_array_lower_histogram_stretching
    sceneRadiance_Lower[:, :, 1] = G_array_lower_histogram_stretching
    sceneRadiance_Lower[:, :, 2] = R_array_lower_histogram_stretching
    sceneRadiance_Lower = np.uint8(sceneRadiance_Lower)

    sceneRadiance_Upper = np.zeros((height, width, 3))
    sceneRadiance_Upper[:, :, 0] = B_array_upper_histogram_stretching
    sceneRadiance_Upper[:, :, 1] = G_array_upper_histogram_stretching
    sceneRadiance_Upper[:, :, 2] = R_array_upper_histogram_stretching
    sceneRadiance_Upper = np.uint8(sceneRadiance_Upper)

    return sceneRadiance_Lower, sceneRadiance_Upper




# def rayleighStr(r, height, width):
#     alpha = 0.4
#     selectedRange = [0, 255]
#     NumPixel = np.zeros(256)
#     temp = np.zeros(256)
#     for i in range(0, height):
#         for j in range(0, width):
#             NumPixel[r[i, j]] = NumPixel[r[i, j]] + 1
#     ProbPixel = NumPixel / (height * width)
#     CumuPixel = np.cumsum(ProbPixel)
#     valSpread = selectedRange[1] - selectedRange[0]
#     hconst = 2 * alpha ** 2
#     vmax = 1 - e ** (-1 / hconst)
#     val = vmax * (CumuPixel)
#     val = np.array(val)
#
#     for i in range(256):
#         if (val[i] >= 1):
#             val[i] = val[i] - esp
#     for i in range(256):
#         temp[i] = np.sqrt(-hconst * math.log((1 - val[i]), e))
#         normalization = temp[i] * valSpread
#         if(normalization > 255):
#             CumuPixel[i] = 255
#         else:
#             CumuPixel[i] = normalization
#     for i in range(0, height):
#         for j in range(0, width):
#             r[i, j] = CumuPixel[(r[i, j])]
#     return r


