# Albumentations
[![Build Status](https://travis-ci.org/albu/albumentations.svg?branch=master)](https://travis-ci.org/albu/albumentations)
[![Documentation Status](https://readthedocs.org/projects/albumentations/badge/?version=latest)](https://albumentations.readthedocs.io/en/latest/?badge=latest)


* The library is faster than other libraries on most of the transformations.
* Based on numpy, OpenCV, imgaug picking the best from each of them.
* Simple, flexible API that allows the library to be used in any computer vision pipeline.
* Large, diverse set of transformations.
* Easy to extend the library to wrap around other libraries.
* Easy to extend to other tasks.
* Supports transformations on images, masks, key points and bounding boxes.
* Supports python 2.7-3.7
* Easy integration with PyTorch.
* Easy transfer from torchvision.
* Was used to get top results in many DL competitions at Kaggle, topcoder, CVPR, MICCAI.
* Written by Kaggle Masters.

## How to use

**All in one showcase notebook** - [`showcase.ipynb`](https://github.com/albu/albumentations/blob/master/notebooks/showcase.ipynb)

**Classification** - [`example.ipynb`](https://github.com/albu/albumentations/blob/master/notebooks/example.ipynb)

**Object detection** - [`example_bboxes.ipynb`](https://github.com/albu/albumentations/blob/master/notebooks/example_bboxes.ipynb)

**Non-8-bit images** - [`example_16_bit_tiff.ipynb`](https://github.com/albu/albumentations/blob/master/notebooks/example_16_bit_tiff.ipynb)

**Image segmentation** [`example_kaggle_salt.ipynb`](https://github.com/albu/albumentations/blob/master/notebooks/example_kaggle_salt.ipynb)

**Custom tasks** such as autoencoders, more then three channel images - refer to `Compose` class [documentation](https://albumentations.readthedocs.io/en/latest/api/core.html#albumentations.core.composition.Compose) to use `additional_targets`.

You can use this [Google Colaboratory notebook](https://colab.research.google.com/drive/1JuZ23u0C0gx93kV0oJ8Mq0B6CBYhPLXy#scrollTo=GwFN-In3iagp&forceEdit=true&offline=true&sandboxMode=true)
to adjust image augmentation parameters and see the resulting images.

![parrot](https://habrastorage.org/webt/bd/ne/rv/bdnerv5ctkudmsaznhw4crsdfiw.jpeg)

![inria](https://habrastorage.org/webt/su/wa/np/suwanpeo6ww7wpwtobtrzd_cg20.jpeg)

![medical](https://habrastorage.org/webt/1i/fi/wz/1ifiwzy0lxetc4nwjvss-71nkw0.jpeg)

![vistas](https://habrastorage.org/webt/rz/-h/3j/rz-h3jalbxic8o_fhucxysts4tc.jpeg)

## Authors
[Alexander Buslaev](https://www.linkedin.com/in/al-buslaev/)

[Alex Parinov](https://www.linkedin.com/in/alex-parinov/)

[Vladimir I. Iglovikov](https://www.linkedin.com/in/iglovikov/)

[Evegene Khvedchenya](https://www.linkedin.com/in/cvtalks/)

## Installation

### PyPI
You can use pip to install albumentations:
```
pip install albumentations
```

If you want to get the latest version of the code before it is released on PyPI you can install the library from GitHub:
```
pip install -U git+https://github.com/albu/albumentations
```

### Conda
To install albumentations using conda we need first to install `imgaug` with pip
```
pip install imgaug
conda install albumentations -c albumentations
```

## Documentation
The full documentation is available at [albumentations.readthedocs.io](https://albumentations.readthedocs.io/en/latest/).


## Migrating from torchvision to albumentations

Migrating from torchvision to albumentations is simple - you just need to change a few lines of code.
Albumentations has equivalents for common torchvision transforms as well as plenty of transforms that are not presented in torchvision.
[`migrating_from_torchvision_to_albumentations.ipynb`](https://github.com/albu/albumentations/blob/master/notebooks/migrating_from_torchvision_to_albumentations.ipynb) shows how one can migrate code from torchvision to albumentations.


## Benchmarking results
To run the benchmark yourself follow the instructions in [benchmark/README.md](https://github.com/albu/albumentations/blob/master/benchmark/README.md)

Results for running the benchmark on first 2000 images from the ImageNet validation set using an Intel Core i7-7800X CPU.
The table shows how many images per second can be processed on a single core, higher is better.


|  | albumentations <br><small>0.1.11</small> | imgaug <br><small>0.2.67</small> | torchvision (Pillow backend) <br><small>0.2.1</small>  | torchvision (Pillow-SIMD backend) <br><small>0.2.1</small> | Keras <br><small>2.2.4</small> | Augmentor <br><small>0.2.3</small> | solt <br><small>0.1.3</small> |
|--------------------|:--------------:|:------:|:-----------:|:-------------------------:|:-----:|:---------:|:----:|
| RandomCrop64 | **754387** | 6730 | 94557 | 97446 | - | 69562 | 7932 |
| PadToSize512 | **7516** | - | 798 | 772 | - | - | 3102 |
| Resize512 | **2898** | 1272 | 379 | 1441 | - | 378 | 1822 |
| HorizontalFlip | 1093 | 1008 | **6475** | 5972 | 1093 | 6346 | 1154 |
| VerticalFlip | **11048** | 5429 | 7845 | 8213 | 10760 | 7677 | 3823 |
| Rotate | **1079** | 772 | 124 | 206 | 37 | 52 | 267 |
| ShiftScaleRotate | **2198** | 1223 | 107 | 184 | 40 | - | - |
| Brightness | 772 | **884** | 425 | 563 | 199 | 425 | 134 |
| Contrast | 894 | 826 | 304 | 401 | - | 303 | **1028** |
| BrightnessContrast | **690** | 408 | 173 | 229 | - | 173 | 119 |
| ShiftHSV | **216** | 151 | 57 | 74 | - | - | 142 |
| ShiftRGB | 728 | **884** | - | - | 665 | - | - |
| Gamma | 1151 | - | 1655 | **1692** | - | - | 918 |
| Grayscale | 2710 | 509 | 1183 | 1515 | - | 2891 | **3872** |


Python and library versions: Python 3.6.8 | Anaconda, numpy 1.16.1, pillow 5.4.1, pillow-simd 5.3.0.post0, opencv-python 4.0.0.21, scikit-image 0.14.2, scipy 1.2.0.


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
4. Run flake8 to perform PEP8 and PEP257 style checks and to check code for lint errors.
   ```
   flake8
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

# Citing

If you find this library useful for your research, please consider citing:

```
@article{2018arXiv180906839B,
    author = {A. Buslaev, A. Parinov, E. Khvedchenya, V.~I. Iglovikov and A.~A. Kalinin},
     title = "{Albumentations: fast and flexible image augmentations}",
   journal = {ArXiv e-prints},
    eprint = {1809.06839},
      year = 2018      
}
```
