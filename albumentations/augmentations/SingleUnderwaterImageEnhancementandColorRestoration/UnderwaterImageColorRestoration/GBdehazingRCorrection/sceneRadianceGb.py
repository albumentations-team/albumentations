import numpy as np


def sceneRadianceGB(img,transmission,AtomsphericLight):
    sceneRadiance = img.copy()
    img = np.float32(img)
    for i in range(0, 2):
        sceneRadiance[:, :, i] = (img[:, :, i] - AtomsphericLight[i]) / transmission[:, :, i] + AtomsphericLight[i]
        # 限制透射率 在0～255
    sceneRadiance = (sceneRadiance - sceneRadiance.min()) / (sceneRadiance.max() - sceneRadiance.min()) * 255
    sceneRadiance = np.clip(sceneRadiance, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)

    return sceneRadiance


