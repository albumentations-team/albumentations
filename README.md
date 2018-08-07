# Albumentations
[![Build Status](https://travis-ci.org/albu/albumentations.svg?branch=master)](https://travis-ci.org/albu/albumentations)
[![Documentation Status](https://readthedocs.org/projects/albumentations/badge/?version=latest)](https://albumentations.readthedocs.io/en/latest/?badge=latest)

* Great fast augmentations based on highly-optimized OpenCV library
* Super simple yet powerful interface for different tasks like (segmentation, detection, etc.)
* Easy to customize
* Easy to add other frameworks

![Vladimir_Iglovikov](https://habrastorage.org/webt/_e/xe/8a/_exe8adren79a0ctavaiq4jf2jo.jpeg)

## Authors
[Alexander Buslaev](https://www.linkedin.com/in/al-buslaev/)

[Alex Parinov](https://www.linkedin.com/in/alex-parinov/)

[Vladimir Iglovikov](https://www.linkedin.com/in/iglovikov/)

## Example usage

```python
from albumentations import (
    CLAHE, RandomRotate90, Transpose, ShiftScaleRotate, Blur, OpticalDistortion, 
    GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, 
    MedianBlur, IAAPiecewiseAffine, IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, 
    Flip, OneOf, Compose
)
import numpy as np

def strong_aug(p=0.5):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
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

image = np.ones((300, 300, 3), dtype=np.uint8)
mask = np.ones((300, 300), dtype=np.uint8)
whatever_data = "my name"
augmentation = strong_aug(p=0.9)
data = {"image": image, "mask": mask, "whatever_data": whatever_data, "additional": "hello"}
augmented = augmentation(**data)
image, mask, whatever_data, additional = augmented["image"], augmented["mask"], augmented["whatever_data"], augmented["additional"]
```

See [`example.ipynb`](notebooks/example.ipynb)


## Installation
You can use pip to install albumentations:
```
pip install albumentations
```

If you want to get the latest version of the code before it is released on PyPI you can install the library from GitHub:
```
pip install -U git+https://github.com/albu/albumentations
```


## Documentation
The full documentation is available at [albumentations.readthedocs.io](https://albumentations.readthedocs.io/en/latest/).


## Demo
You can use this [Google Colaboratory notebook](https://colab.research.google.com/drive/1JuZ23u0C0gx93kV0oJ8Mq0B6CBYhPLXy#scrollTo=GwFN-In3iagp&forceEdit=true&offline=true&sandboxMode=true)
to adjust image augmentation parameters and see the resulting images.


## Working with non-8-bit images
[`example_16_bit_tiff.ipynb`](notebooks/example_16_bit_tiff.ipynb) shows how albumentations can be used to work with non-8-bit images (such as 16-bit and 32-bit TIFF images).


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


## Comments
In some systems, in the multiple GPU regime PyTorch may deadlock the DataLoader if OpenCV was compiled with OpenCL optimizations. Adding the following two lines before the library import may help. For more details [https://github.com/pytorch/pytorch/issues/1355](https://github.com/pytorch/pytorch/issues/1355) 

```python
cv2.setNumThreads(0)	
cv2.ocl.setUseOpenCL(False)
```

### Thanks:
Special thanks to [@creafz](https://github.com/creafz) for refactoring, documentation, tests, CI and benchmarks. Awesome work!
