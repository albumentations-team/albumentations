Image only transforms
=====================

MultiplicativeNoise
-------------------

API link: :class:`~albumentations.augmentations.transforms.MultiplicativeNoise`


1. Original image
2. :code:`MultiplicativeNoise(multiplier=0.5, p=1)`
3. :code:`MultiplicativeNoise(multiplier=1.5, p=1)`
4. :code:`MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True, p=1)`
5. :code:`MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, p=1)`
6. :code:`MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, per_channel=True, p=1)`

.. figure:: ./images/MultiplicativeNoise.jpg
    :alt: MultiplicativeNoise image


ToSepia
-------------------

API link: :class:`~albumentations.augmentations.transforms.ToSepia`


1. Original image
2. :code:`ToSepia(p=1)`

.. figure:: ./images/ToSepia.jpg
    :alt: ToSepia image
