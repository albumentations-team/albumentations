import numpy as np


def sceneRadianceRGB(img, transmissionB, transmissionG, transmissionR, AtomsphericLight):
    transmission = np.zeros(img.shape)
    transmission[:, :, 0] = transmissionB
    transmission[:, :, 1] = transmissionG
    transmission[:, :, 2] = transmissionR
    sceneRadiance = np.zeros(img.shape)
    img = np.float32(img)
    for i in range(0, 3):
        sceneRadiance[:, :, i] = (img[:, :, i] - AtomsphericLight[i]) / transmission[:, :, i]  + AtomsphericLight[i]
        # 限制透射率 在0～255
    sceneRadiance = np.clip(sceneRadiance,0, 255)
    sceneRadiance = np.uint8(sceneRadiance)
    return sceneRadiance


