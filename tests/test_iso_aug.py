import random

import cv2
import numpy as np

import albumentations as A
from albumentations.augmentations.functional import clipped


@clipped
def iso_noise(image, intensity=(0, 10), coverage=0.2, **kwargs):
    noise = np.zeros_like(image)
    for i in range(noise.shape[0]):
        for j in range(noise.shape[1]):
            for c in range(noise.shape[2]):
                noise[i, j, c] = random.uniform(intensity[0], intensity[1]) if random.uniform(0, 1) < coverage else 0

    dampen_factor = 1.0 - image / 255.
    return image + (noise * dampen_factor).astype(image.dtype)


class ISONoise(A.ImageOnlyTransform):
    def __init__(self, intensity=(10, 30), coverage=0.2, p=0.5):
        super().__init__(p=p)
        self.intensity = intensity
        self.coverage = coverage

    def apply(self, img, intensity=(10, 30), coverage=0.2, **params):
        print(intensity, coverage)
        return iso_noise(img, intensity, coverage)

    def get_transform_init_args_names(self):
        return ('intensity', 'coverage')

    def get_params(self):
        return {
            'intensity': self.intensity,
            'coverage': self.coverage
        }


def test_aug():
    aug = ISONoise(p=1)

    while True:
        # image = np.zeros((512, 512, 3), dtype=np.uint8) + 40
        image = cv2.imread('monkey.jpg')
        image = aug(image=image)['image']
        cv2.imshow('image',image)
        cv2.waitKey(-1)