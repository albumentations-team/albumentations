import numpy as np

def  sceneradiance(img, sceneRadiance_GB):
    img = np.float32(img)
    R_original = img[:, :, 2]
    sceneRadiance_GB = np.float32(sceneRadiance_GB)

    print('***********************************************************')
    avgRr = 1.5 - (np.mean(sceneRadiance_GB[:,:,0])/255 +np.mean(sceneRadiance_GB[:,:,1])/255)
    parameterR  =   avgRr  / ((np.mean(R_original))/255)

    print('parameterR',parameterR)
    sceneRadianceR = R_original * parameterR
    sceneRadianceR = (sceneRadianceR - sceneRadianceR.min()) / (sceneRadianceR.max() - sceneRadianceR.min())
    sceneRadianceR = sceneRadianceR * 255

    sceneRadianceR = np.clip(sceneRadianceR, 0, 255)
    sceneRadiance_GB[:, :, 2] = sceneRadianceR
    sceneRadiance_GB = np.uint8(sceneRadiance_GB)
    return  sceneRadiance_GB



