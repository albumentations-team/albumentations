import numpy as np

def transmissionMap(transmissionR, AtomsphericLight,BG ,AtomsphericLight_R, R):
    x = AtomsphericLight_R * ( -0.00113 * BG + 1.62517)
    y = AtomsphericLight * ( -0.00113 * R + 1.62517)
    k = x/y
    transmission = transmissionR ** k
    transmission = np.clip(transmission, 0.1, 1)
    return transmission

def getGBTransmissionESt(transmissionR, AtomsphericLight):
    transmissionB  = transmissionMap(transmissionR, AtomsphericLight[0],450 , AtomsphericLight[2], 620)
    transmissionG  = transmissionMap(transmissionR, AtomsphericLight[1],540 , AtomsphericLight[2], 620)

    return transmissionB,transmissionG