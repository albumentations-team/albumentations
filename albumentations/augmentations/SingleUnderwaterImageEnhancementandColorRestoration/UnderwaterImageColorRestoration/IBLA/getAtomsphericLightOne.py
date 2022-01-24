import numpy as np

# 用于排序时存储原来像素点位置的数据结构
class Node(object):
	def __init__(self,x,y,value):
		self.x = x
		self.y = y
		self.value = value
	def printInfo(self):
		print(self.x,self.y,self.value)


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

