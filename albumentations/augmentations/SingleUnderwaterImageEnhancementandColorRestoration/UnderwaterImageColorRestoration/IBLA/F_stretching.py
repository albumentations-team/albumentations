import numpy as np

def StretchingFusion(map):
    map_max = np.max(map)
    map_min = np.min(map)

    # if(map_max < 2):
    #     map_max = 5
    finalmap  = (map - map_min) / (map_max - map_min)
    return finalmap

