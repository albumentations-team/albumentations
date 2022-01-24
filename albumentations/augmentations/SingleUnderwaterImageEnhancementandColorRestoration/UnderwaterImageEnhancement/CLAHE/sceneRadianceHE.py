import cv2


def RecoverHE(sceneRadiance):
    # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(2, 2))
    for i in range(3):
        sceneRadiance[:, :, i] =  cv2.equalizeHist(sceneRadiance[:, :, i])
        # sceneRadiance[:, :, i] = clahe.apply((sceneRadiance[:, :, i]))
    return sceneRadiance