import numpy as np

def cal_equalisation(img,ratio):
    Array = img * ratio
    Array = np.clip(Array, 0, 255)
    return Array

def RGB_equalisation(img):
    img = np.float32(img)
    avg_RGB = []
    for i in range(3):
        avg = np.mean(img[:,:,i])
        avg_RGB.append(avg)
    avg_RGB = 128/np.array(avg_RGB)
    ratio = avg_RGB

    for i in range(0,2):
        img[:,:,i] = cal_equalisation(img[:,:,i],ratio[i])
    return img
