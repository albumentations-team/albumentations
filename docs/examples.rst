Examples
========

.. toctree::
   :maxdepth: 2

.. code-block:: python

   from albumentations import (
       HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
       Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
       IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
       IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose
   )
   import numpy as np

   def strong_aug(p=.5):
       return Compose([
           RandomRotate90(),
           Flip(),
           Transpose(),
           OneOf([
               IAAAdditiveGaussianNoise(),
               GaussNoise(),
           ], p=0.2),
           OneOf([
               MotionBlur(p=.2),
               MedianBlur(blur_limit=3, p=.1),
               Blur(blur_limit=3, p=.1),
           ], p=0.2),
           ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
           OneOf([
               OpticalDistortion(p=0.3),
               GridDistortion(p=.1),
               IAAPiecewiseAffine(p=0.3),
           ], p=0.2),
           OneOf([
               CLAHE(clip_limit=2),
               IAASharpen(),
               IAAEmboss(),
               RandomContrast(),
               RandomBrightness(),
           ], p=0.3),
           HueSaturationValue(p=0.3),
       ], p=p)

   image = np.ones((300, 300))
   mask = np.ones((300, 300))
   whatever_data = "my name"
   augmentation = strong_aug(p=0.9)
   data = {"image": image, "mask": mask, "whatever_data": whatever_data, "additional": "hello"}
   augmented = augmentation(**data)
   image, mask, whatever_data, additional = augmented["image"], augmented["mask"], augmented["whatever_data"], augmented["additional"]


For more examples see `example.ipynb <https://github.com/albu/albumentations/blob/master/example.ipynb>`_
