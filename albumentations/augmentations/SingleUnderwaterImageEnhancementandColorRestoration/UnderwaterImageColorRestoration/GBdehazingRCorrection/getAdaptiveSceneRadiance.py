import numpy as np

def AdaptiveSceneRadiance(sceneRadiance,S_x):
    sceneRadiance = np.float64(sceneRadiance)
    sceneRadiance  = sceneRadiance * S_x

    sceneRadiance = np.clip(sceneRadiance, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)
    return sceneRadiance