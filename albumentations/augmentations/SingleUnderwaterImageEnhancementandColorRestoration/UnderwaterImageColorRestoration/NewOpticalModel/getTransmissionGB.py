import cv2
import numpy as np
def TransmissionGB(sactterRate):
    transmissionGB  = 1 - sactterRate
    # transmissionGB = np.clip(transmissionGB, 0.1, 0.9)
    transmission = transmissionGB
    return transmission




