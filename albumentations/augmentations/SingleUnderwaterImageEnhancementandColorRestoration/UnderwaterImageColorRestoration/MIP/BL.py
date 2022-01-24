import numpy as np

# 用于排序时存储原来像素点位置的数据结构
class Node(object):
	def __init__(self,x,y,value):
		self.x = x
		self.y = y
		self.value = value
	def printInfo(self):
		print(self.x,self.y,self.value)


def getAtomsphericLight(transmission, img):
    # print('transmission',transmission)
    height = len(transmission)
    width = len(transmission[0])
    nodes = []
    # print('height*width',height*width)
    # 用一个链表结构(list)存储数据
    for i in range(0, height):
        for j in range(0, width):
            oneNode = Node(i, j, transmission[i, j])
            nodes.append(oneNode)
    # 排序
    nodes = sorted(nodes, key=lambda node: node.value, reverse=False)
    atomsphericLight  = img[nodes[0].x, nodes[0].y, :]
    return atomsphericLight

