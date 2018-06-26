# Albumentations
[![Build Status](https://travis-ci.org/albu/albumentations.svg?branch=master)](https://travis-ci.org/albu/albumentations)
[![Documentation Status](https://readthedocs.org/projects/albumentations/badge/?version=latest)](https://albumentations.readthedocs.io/en/latest/?badge=latest)

* Great fast augmentations based on highly-optimized OpenCV library
* Super simple yet powerful interface for different tasks like (segmentation, detection, etc.)
* Easy to customize
* Easy to add other frameworks

## Example usage:

```python
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
```

See `example.ipynb`

## Installation
You can use pip to install the latest version from GitHub:
```
pip install -U git+https://github.com/albu/albumentations
```

## Documentation
The full documentation is available at [albumentations.readthedocs.io](https://albumentations.readthedocs.io/en/latest/).


## Contributing
1. Clone the repository:
```
git clone git@github.com:albu/albumentations.git
cd albumentations
```
2. Install the library in development mode:
```
pip install -e .[tests]
```
3. Run tests:
```
pytest
```

## Building the documentation
1. Go to `docs/` directory
```
cd docs
```
2. Install required libraries
```
pip install -r requirements.txt
```
3. Build html files
```
make html
```
4. Open `_build/html/index.html` in browser.

Alternatively, you can start a web server that rebuilds the documentation
automatically when a change is detected by running `make livehtml`


### Roadmap:
 - [ ] add tests
 - [ ] benchmark 
 - [ ] add comments and dockstrings
 - [ ] add LICENSE
