import numpy as np

#
# # ----------------------------RGHS--------------------------------
# def cal_equalisation(array, a,height,width):
#     Array = array * a
#     for i in range(height):
#         for j in range(width):
#             if(Array[i][j]>255):
#                 Array[i][j] = 255
#             elif(Array[i][j]<0):
#                 Array[i][j] = 0
#             else:
#                 pass
#     return Array
#
# def RGB_equalisation(r, g, b,height,width):
#     float_r = r.astype(np.float64)
#     float_g = g.astype(np.float64)
#     float_b = b.astype(np.float64)
#     r_avg  = np.mean(float_r)
#     g_avg  = np.mean(float_g)
#     b_avg  = np.mean(float_b)
#     a_r = 128/r_avg
#     a_g = 128/g_avg
#     a_b = 128/b_avg
#     # float_r = cal_equalisation(float_r, a_r, height, width)
#     # float_g = cal_equalisation(float_g,a_g,height,width)
#     # float_b = cal_equalisation(float_b,a_b,height,width)
#     return float_r,float_g, float_b



# ---------------------------------UCM----------------------------------

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
    # print('avg_RGB',avg_RGB)
    a_r = avg_RGB[0]/avg_RGB[2]
    a_g =  avg_RGB[0]/avg_RGB[1]
    ratio = [0,a_g,a_r]
    for i in range(1,3):
        img[:,:,i] = cal_equalisation(img[:,:,i],ratio[i])
    return img



