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

|                  | albumentations <br><small>0.1.2</small>| imgaug<br><small>0.2.6</small> | torchvision <br>(Pillow backend)<br><small>0.2.1</small> | torchvision<br>(Pillow-SIMD backend)<br><small>0.2.1</small>  | Keras<br><small>2.2.4</small>  |
|------------------|:--------------:|:------:|:--------------------------------:|:------------------------------------:|:-----:|
| RandomCrop64     | **746838**     | -      | 98793                            | 100889                               | -     |
| PadToSize512     | **8270**       | -      | 759                              | 823                                  | -     |
| HorizontalFlip   | 1319           | 938    | 6307                             | **6495**                             | 1025  |
| VerticalFlip     | **11362**      | 5005   | 8545                             | 8651                                 | 11108 |
| Rotate           | **1084**       | 786    | 123                              | 210                                  | 37    |
| ShiftScaleRotate | **1993**       | 1228   | 107                              | 188                                  | 40    |
| Brightness       | **896**        | 841    | 426                              | 567                                  | 199   |
| ShiftHSV         | **219**        | 144    | 57                               | 73                                   | -     |
| ShiftRGB         | 725            |**900** | -                                | -                                    | 663   |
| Gamma            | 1354           | -      | **1724**                         | 1713                                 | -     |
| Grayscale        | **2603**       | 330    | 1145                             | 1544                                 | -     |


Python and library versions: Python 3.6.6 | Anaconda, numpy 1.15.2, pillow 5.3.0, pillow-simd 5.2.0.post0, opencv-python 3.4.3.18, scikit-image 0.14.1, scipy 1.1.0.


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
