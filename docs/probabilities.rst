About probabilities.
=====================

.. toctree::
   :maxdepth: 2


Default probability values
**************************

All pre / post processing transforms: **Compose**, **PadIfNeeded**, **CenterCrop**, **RandomCrop**, **Crop**, **RandomCropNearBBox**, **RandomSizedCrop**, **RandomResizedCrop**, **RandomSizedBBoxSafeCrop**, **CropNonEmptyMaskIfExists**, **Lambda**,  **Normalize**, **ToFloat**, **FromFloat**, **ToTensor**, **LongestMaxSize** have default
probability values equal to **1**. All others are equal to **0.5**


.. code-block:: python

    from albumentations import (
       RandomRotate90, IAAAdditiveGaussianNoise, GaussNoise, Compose, OneOf
    )
    import numpy as np

    def aug(p1, p2, p3):
       return Compose([
           RandomRotate90(p=p2),
           OneOf([
               IAAAdditiveGaussianNoise(p=0.9),
               GaussNoise(p=0.6),
           ], p=p3)
       ], p=p1)

    image = np.ones((300, 300, 3), dtype=np.uint8)
    mask = np.ones((300, 300), dtype=np.uint8)
    whatever_data = "my name"
    augmentation = aug(p1=0.9, p2=0.7, p3=0.3)
    data = {"image": image, "mask": mask, "whatever_data": whatever_data, "additional": "hello"}
    augmented = augmentation(**data)
    image, mask, whatever_data, additional = augmented["image"], augmented["mask"], augmented["whatever_data"], augmented["additional"]




In the above augmentation pipeline, we have three types of probabilities. Combination of them is the primary factor that
decides how often each of them will be applied.

 1. **p1**: decides if this augmentation will be applied. The most common case is **p1=1** means that we always apply the transformations from above. **p1=0** will mean that the transformation block will be ignored.
 2. **p2**: every augmentation has an option to be applied with some probability.
 3. **p3**: decide if **OneOf** will be applied.

OneOf Block
*************

To decide which augmentation within **OneOf** block is used the following rule is applied.

 1. We normalize all probabilities within a block to one.
 After this we pick augmentation based on the normalized probabilities.
 In the example above **IAAAdditiveGaussianNoise** has probability **0.9** and **GaussNoise** probability **0.6**.
 After normalization, they become **0.6** and **0.4**.
 Which means that we decide if we should use **IAAAdditiveGaussianNoise** with probability **0.6** and **GaussNoise** otherwise.
 2. If we picked to consider **GaussNoise** the next step we call **GaussNoise** with flag **force_apply=True**.

Example calculations
********************
Thus, each augmentation in the example above will be applied with the probability:

 1. **RandomRotate90**: `p1 * p2`
 2. **IAAAdditiveGaussianNoise**: `p1 * p3 * (0.9 / (0.9 + 0.6))`
 3. **GaussianNoise**: `p1 * p3 * (0.6 / (0.9 + 0.6))`
