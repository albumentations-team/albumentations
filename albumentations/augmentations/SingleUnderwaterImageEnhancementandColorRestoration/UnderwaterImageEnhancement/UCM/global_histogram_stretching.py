import numpy as np

def histogram_r(r_array,height, width):
    length = height * width
    R_rray = []
    for i in range(height):
        for j in range(width):
            R_rray.append(r_array[i][j])
    R_rray.sort()
    I_min = int(R_rray[int(length / 500)])
    I_max = int(R_rray[-int(length / 500)])
    array_Global_histogram_stretching = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            if r_array[i][j] < I_min:
                # p_out = r_array[i][j]
                array_Global_histogram_stretching[i][j] = I_min
            elif (r_array[i][j] > I_max):
                p_out = r_array[i][j]
                array_Global_histogram_stretching[i][j] = 255
            else:
                p_out = int((r_array[i][j] - I_min) * ((255 - I_min) / (I_max - I_min)))+ I_min
                array_Global_histogram_stretching[i][j] = p_out
    return (array_Global_histogram_stretching)

def histogram_g(r_array,height, width):
    length = height * width
    R_rray = []
    for i in range(height):
        for j in range(width):
            R_rray.append(r_array[i][j])
    R_rray.sort()
    I_min = int(R_rray[int(length / 500)])
    I_max = int(R_rray[-int(length / 500)])
    array_Global_histogram_stretching = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            if r_array[i][j] < I_min:
                p_out = r_array[i][j]
                array_Global_histogram_stretching[i][j] = 0
            elif (r_array[i][j] > I_max):
                p_out = r_array[i][j]
                array_Global_histogram_stretching[i][j] = 255
            else:
                p_out = int((r_array[i][j] - I_min) * ((255) / (I_max - I_min)) )
                array_Global_histogram_stretching[i][j] = p_out
    return (array_Global_histogram_stretching)

def histogram_b(r_array,height, width):
    length = height * width
    R_rray = []
    for i in range(height):
        for j in range(width):
            R_rray.append(r_array[i][j])
    R_rray.sort()
    I_min = int(R_rray[int(length / 500)])
    I_max = int(R_rray[-int(length / 500)])
    array_Global_histogram_stretching = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            if r_array[i][j] < I_min:
                # p_out = r_array[i][j]
                array_Global_histogram_stretching[i][j] = 0
            elif (r_array[i][j] > I_max):
                # p_out = r_array[i][j]
                array_Global_histogram_stretching[i][j] = I_max
            else:
                p_out = int((r_array[i][j] - I_min) * ((I_max) / (I_max - I_min)))
                array_Global_histogram_stretching[i][j] = p_out
    return (array_Global_histogram_stretching)

def stretching(img):
    height = len(img)
    width = len(img[0])
    img[:, :, 2] = histogram_r(img[:, :, 2], height, width)
    img[:, :, 1] = histogram_g(img[:, :, 1], height, width)
    img[:, :, 0] = histogram_b(img[:, :, 0], height, width)
    return img


