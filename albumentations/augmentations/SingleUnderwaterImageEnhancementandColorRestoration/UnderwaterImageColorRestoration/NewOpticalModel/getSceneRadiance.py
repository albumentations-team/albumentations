import numpy as np


def SceneRadiance(img,transmissionGB,transmissionR,sactterRate,AtomsphericLight):
    img = np.float16(img)
    transmission = np.zeros(img.shape)
    transmission[:, :, 0] = transmissionGB
    transmission[:, :, 1] = transmissionGB
    transmission[:, :, 2] = transmissionR
    # print('transmission',transmission)
    sceneRadiance = np.zeros(img.shape)
    for i in range(0, 3):
        sceneRadiance[:, :, i] = (img[:, :, i] - sactterRate * AtomsphericLight[i]) / transmission[:, :, i]
        # sceneRadiance[:, :, i] = (img[:, :, i] -  AtomsphericLight[i]) / transmission[:, :, i] + AtomsphericLight[i]
    sceneRadiance = np.clip(sceneRadiance,0, 255)
    sceneRadiance = np.uint8(sceneRadiance)
    return sceneRadiance


