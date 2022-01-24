import numpy as np

def getTransmission(largestDiff):
    transmission = largestDiff + (1 - np.max(largestDiff))

    transmission = np.clip(transmission, 0, 1)

    return transmission