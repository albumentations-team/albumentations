import numpy as np

def global_stretching(img_L,height, width):
    length = height * width
    R_rray = (np.copy(img_L)).flatten()
    R_rray.sort()
    print('R_rray',R_rray)
    I_min = int(R_rray[int(length / 100)])
    I_max = int(R_rray[-int(length / 100)])
    print('I_min',I_min)
    print('I_max',I_max)
    array_Global_histogram_stretching_L = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            if img_L[i][j] < I_min:
                p_out = img_L[i][j]
                array_Global_histogram_stretching_L[i][j] = 0
            elif (img_L[i][j] > I_max):
                p_out = img_L[i][j]
                array_Global_histogram_stretching_L[i][j] = 100
            else:
                p_out = int((img_L[i][j] - I_min) * ((100) / (I_max - I_min)))
                array_Global_histogram_stretching_L[i][j] = p_out
    return (array_Global_histogram_stretching_L)

