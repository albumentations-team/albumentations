import cv2
import numpy as np


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


def getAtomsphericLightLv(img):
    I_gray_Q = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(0,5):
        img,I_gray_Q  = quadTree(img,I_gray_Q)
    AtomsphericLight = np.zeros(3)
    for i in range(3):
        AtomsphericLight[i] = np.mean(img[:,:,i])
    return AtomsphericLight


