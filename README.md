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


## Benchmarking results
To run the benchmark yourself follow the instructions in [benchmark/README.md](benchmark/README.md)

Results for running the benchmark on first 2000 images from the ImageNet validation set using an Intel Core i7-7800X CPU. All times are in seconds, lower is better.

|                   | albumentations  |  imgaug  | torchvision<br> (Pillow backend)| torchvision<br> (Pillow-SIMD backend) |  Keras   |
|-------------------|:---------------:|:--------:|:-------------------------------:|:-------------------------------------:|:--------:|
| RandomCrop64      |    **0.0017**   |    -     |             0.0182              |               0.0182                  |    -     |
| PadToSize512      |    **0.2413**   |    -     |             2.493               |               2.3682                  |    -     |
| HorizontalFlip    |     0.7765      |  2.2299  |           **0.3031**            |               0.3054                  |  2.0508  |
| VerticalFlip      |    **0.178**    |  0.3899  |             0.2326              |               0.2308                  |  0.1799  |
| Rotate            |    **3.8538**   |  4.0581  |             16.16               |               9.5011                  | 50.8632  |
| ShiftScaleRotate  |    **2.0605**   |  2.4478  |            18.5401              |              10.6062                  | 47.0568  |
| Brightness        |    **2.1018**   |  2.3607  |             4.6854              |               3.4814                  |  9.9237  |
| ShiftHSV          |    **10.3925**  | 14.2255  |            34.7778              |              27.0215                  |    -     |
| ShiftRGB          |     2.6159      |**2.1989**|               -                 |                 -                     |  3.0598  |
| Gamma             |     1.4832      |    -     |            **1.1397**           |               1.1447                  |    -     |
| Grayscale         |    **1.2048**   |  5.3895  |             1.6826              |               1.2721                  |    -     |


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


### Thanks:
Special thanks to [@creafz](https://github.com/creafz) for refactoring, documentation, tests, CI and benchmarks. Awesome work!
