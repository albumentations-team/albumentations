from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


def gaussian_smooth(arr, sigma=1):
    n = len(arr)
    _range = range(-int(n/2),int(n/2))
    gauassian = np.array([1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-float(x)**2/(2*sigma**2)) for x in _range])
    smoothed = np.convolve(arr, gauassian, 'same')
    return smoothed

img = np.zeros((500, 500, 3))


patch_size = 30  # magnitude of the perturbation from the unit circle,

offset_x = 150
offset_y = 150

N = 200
vertices = np.zeros((N, 2))

angles = np.linspace(0, 2 * np.pi, N)
r = patch_size * np.random.lognormal(size=N)

x_points = r * np.cos(angles)
y_points = r * np.sin(angles)

x_points = gaussian_smooth(x_points, sigma=5)
y_points = gaussian_smooth(y_points, sigma=5)

vertices[:, 0] = x_points
vertices[:, 1] = y_points

vertices[:, 0] = vertices[:, 0] + offset_x
vertices[:, 1] = vertices[:, 1] + offset_y

vertices[-1, :] = vertices[0, :]

pts = vertices.reshape((-1, 1, 2)).astype(np.int32)

cv2.fillPoly(img, [pts], color=(0, 0, 255))
plt.imshow(img)
plt.show()
