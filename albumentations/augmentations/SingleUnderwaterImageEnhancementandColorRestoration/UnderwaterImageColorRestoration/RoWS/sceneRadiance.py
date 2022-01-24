import numpy as np

def sceneRadianceRGB(img, transmission, AtomsphericLight):
    AtomsphericLight = np.array(AtomsphericLight)
    img = np.float64(img)
    sceneRadiance = np.zeros(img.shape)
    transmission = np.clip(transmission, 0.1, 0.9)
    for i in range(0, 3):
        sceneRadiance[:, :, i] = (img[:, :, i] - AtomsphericLight[i]) / transmission  + AtomsphericLight[i]

    sceneRadiance = np.clip(sceneRadiance, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)
    return sceneRadiance



