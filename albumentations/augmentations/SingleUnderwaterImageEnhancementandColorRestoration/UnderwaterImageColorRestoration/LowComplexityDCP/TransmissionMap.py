import numpy as np
import cv2
def TransmissionComposition(folder, transmission, number, param):
    transmission = (transmission * 255).astype(np.uint8)
    cv2.imwrite(folder+ 'TansmissionMap/'+ number + param + '_LowComplexityDCP.jpg', transmission)
