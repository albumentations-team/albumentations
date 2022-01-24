import numpy as np

def global_stretching(img_L):
    height = len(img_L)
    width = len(img_L[0])
    length = height * width
    R_rray = []
    for i in range(height):
        for j in range(width):
            R_rray.append(img_L[i][j])
    R_rray.sort()
    I_min = R_rray[int(length / 2000)]
    I_max = R_rray[-int(length / 2000)]
    # print('I_min',I_min)
    # print('I_max',I_max)
    array_Global_histogram_stretching_L = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            if img_L[i][j] < I_min:
                p_out = img_L[i][j]
                array_Global_histogram_stretching_L[i][j] = 0
            elif (img_L[i][j] > I_max):
                p_out = img_L[i][j]
                array_Global_histogram_stretching_L[i][j] = 1
            else:
                p_out = (img_L[i][j] - I_min) * ((1-0) / (I_max - I_min))+ 0
                array_Global_histogram_stretching_L[i][j] = p_out
    return (array_Global_histogram_stretching_L)














# def histogram_r(r_array,height, width):
#     length = height * width
#     R_rray = []
#     for i in range(height):
#         for j in range(width):
#             R_rray.append(r_array[i][j])
#     R_rray.sort()
#     I_min = int(R_rray[int(length / 20000)])
#     I_max = int(R_rray[-int(length / 20000)])
#     array_Global_histogram_stretching = np.zeros((height, width))
#     for i in range(0, height):
#         for j in range(0, width):
#             if r_array[i][j] < I_min:
#                 p_out = r_array[i][j]
#                 array_Global_histogram_stretching[i][j] = p_out
#             elif (r_array[i][j] > I_max):
#                 p_out = r_array[i][j]
#                 array_Global_histogram_stretching[i][j] = p_out
#             else:
#                 p_out = int((r_array[i][j] - I_min) * (
#                     (255) / (I_max - I_min)))
#                 array_Global_histogram_stretching[i][j] = p_out
#     return (array_Global_histogram_stretching)