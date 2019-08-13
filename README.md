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

## Table of contents
- [How to use](#how-to-use)
- [Authors](#authors)
- [Installation](#installation)
  - [PyPI](#pypi)
  - [Conda](#conda)
- [Documentation](#documentation)
- [Pixel-level transforms](#pixel-level-transforms)
- [Spatial-level transforms](#spatial-level-transforms)
- [Migrating from torchvision to albumentations](#migrating-from-torchvision-to-albumentations)
- [Benchmarking results](#benchmarking-results)
- [Contributing](#contributing)
  - [Adding new transforms](#adding-new-transforms)
- [Building the documentation](#building-the-documentation)
- [Comments](#comments)
- [Citing](#citing)

## How to use

**All in one showcase notebook** - [`showcase.ipynb`](https://github.com/albu/albumentations/blob/master/notebooks/showcase.ipynb)

**Classification** - [`example.ipynb`](https://github.com/albu/albumentations/blob/master/notebooks/example.ipynb)

**Object detection** - [`example_bboxes.ipynb`](https://github.com/albu/albumentations/blob/master/notebooks/example_bboxes.ipynb)

**Non-8-bit images** - [`example_16_bit_tiff.ipynb`](https://github.com/albu/albumentations/blob/master/notebooks/example_16_bit_tiff.ipynb)

**Image segmentation** [`example_kaggle_salt.ipynb`](https://github.com/albu/albumentations/blob/master/notebooks/example_kaggle_salt.ipynb)

**Keypoints** [`example_keypoints.ipynb`](https://github.com/albu/albumentations/blob/master/notebooks/example_keypoints.ipynb)

**Custom targets** [`example_multi_target.ipynb`](https://github.com/albu/albumentations/blob/master/notebooks/example_multi_target.ipynb)

**Weather transforms** [`example_weather_transforms.ipynb`](https://github.com/albu/albumentations/blob/master/notebooks/example_weather_transforms.ipynb)

**Serialization** [`serialization.ipynb`](https://github.com/albu/albumentations/blob/master/notebooks/serialization.ipynb)

You can use this [Google Colaboratory notebook](https://colab.research.google.com/drive/1JuZ23u0C0gx93kV0oJ8Mq0B6CBYhPLXy#scrollTo=GwFN-In3iagp&forceEdit=true&offline=true&sandboxMode=true)
to adjust image augmentation parameters and see the resulting images.

![parrot](https://habrastorage.org/webt/bd/ne/rv/bdnerv5ctkudmsaznhw4crsdfiw.jpeg)

![inria](https://habrastorage.org/webt/su/wa/np/suwanpeo6ww7wpwtobtrzd_cg20.jpeg)

![medical](https://habrastorage.org/webt/1i/fi/wz/1ifiwzy0lxetc4nwjvss-71nkw0.jpeg)

![vistas](https://habrastorage.org/webt/rz/-h/3j/rz-h3jalbxic8o_fhucxysts4tc.jpeg)

<img src="https://habrastorage.org/webt/e-/6k/z-/e-6kz-fugp2heak3jzns3bc-r8o.jpeg" width=100%>

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

And it also works in Kaggle GPU kernels [(proof)](https://www.kaggle.com/creafz/albumentations-installation/)
```
!pip install albumentations > /dev/null
```

### Conda
To install albumentations using conda we need first to install `imgaug` via conda-forge collection
```
conda install -c conda-forge imgaug
conda install albumentations -c albumentations
```

## Documentation
The full documentation is available at [albumentations.readthedocs.io](https://albumentations.readthedocs.io/en/latest/).


## Pixel-level transforms
Pixel-level transforms will change just an input image and will leave any additional targets such as masks, bounding boxes, and keypoints unchanged. The list of pixel-level transforms:

- [Blur](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Blur)
- [CLAHE](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.CLAHE)
- [ChannelDropout](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.ChannelDropout)
- [ChannelShuffle](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.ChannelShuffle)
- [CoarseDropout](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.CoarseDropout)
- [Cutout](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Cutout)
- [FromFloat](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.FromFloat)
- [GaussNoise](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.GaussNoise)
- [GaussianBlur](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.GaussianBlur)
- [HueSaturationValue](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.HueSaturationValue)
- [IAAAdditiveGaussianNoise](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.imgaug.transforms.IAAAdditiveGaussianNoise)
- [IAAEmboss](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.imgaug.transforms.IAAEmboss)
- [IAASharpen](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.imgaug.transforms.IAASharpen)
- [IAASuperpixels](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.imgaug.transforms.IAASuperpixels)
- [ISONoise](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.ISONoise)
- [InvertImg](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.InvertImg)
- [JpegCompression](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.JpegCompression)
- [MedianBlur](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.MedianBlur)
- [MotionBlur](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.MotionBlur)
- [Normalize](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Normalize)
- [RGBShift](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RGBShift)
- [RandomBrightness](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomBrightness)
- [RandomBrightnessContrast](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomBrightnessContrast)
- [RandomContrast](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomContrast)
- [RandomFog](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomFog)
- [RandomGamma](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomGamma)
- [RandomRain](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomRain)
- [RandomShadow](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomShadow)
- [RandomSnow](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomSnow)
- [RandomSunFlare](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomSunFlare)
- [Solarize](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Solarize)
- [ToFloat](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.ToFloat)
- [ToGray](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.ToGray)

## Spatial-level transforms
Spatial-level transforms will simultaneously change both an input image as well as additional targets such as masks, bounding boxes, and keypoints. The following table shows which additional targets are supported by each transform.

| Transform                                                                                                                                                         | Image | Masks | BBoxes | Keypoints |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---: | :---: | :----: | :-------: |
| [CenterCrop](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.CenterCrop)                           | ✓     | ✓     | ✓      | ✓         |
| [Crop](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Crop)                                       | ✓     | ✓     | ✓      |           |
| [ElasticTransform](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.ElasticTransform)               | ✓     | ✓     |        |           |
| [Flip](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Flip)                                       | ✓     | ✓     | ✓      | ✓         |
| [GridDistortion](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.GridDistortion)                   | ✓     | ✓     |        |           |
| [HorizontalFlip](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.HorizontalFlip)                   | ✓     | ✓     | ✓      | ✓         |
| [IAAAffine](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.imgaug.transforms.IAAAffine)                                    | ✓     | ✓     | ✓      | ✓         |
| [IAACropAndPad](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.imgaug.transforms.IAACropAndPad)                            | ✓     | ✓     | ✓      | ✓         |
| [IAAFliplr](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.imgaug.transforms.IAAFliplr)                                    | ✓     | ✓     | ✓      | ✓         |
| [IAAFlipud](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.imgaug.transforms.IAAFlipud)                                    | ✓     | ✓     | ✓      | ✓         |
| [IAAPerspective](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.imgaug.transforms.IAAPerspective)                          | ✓     | ✓     | ✓      | ✓         |
| [IAAPiecewiseAffine](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.imgaug.transforms.IAAPiecewiseAffine)                  | ✓     | ✓     | ✓      | ✓         |
| [Lambda](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Lambda)                                   | ✓     | ✓     | ✓      | ✓         |
| [LongestMaxSize](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.LongestMaxSize)                   | ✓     | ✓     | ✓      |           |
| NoOp                                                                                                                                                              | ✓     | ✓     | ✓      | ✓         |
| [OpticalDistortion](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.OpticalDistortion)             | ✓     | ✓     |        |           |
| [PadIfNeeded](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.PadIfNeeded)                         | ✓     | ✓     | ✓      | ✓         |
| [RandomCrop](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomCrop)                           | ✓     | ✓     | ✓      | ✓         |
| [RandomCropNearBBox](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomCropNearBBox)           | ✓     | ✓     | ✓      |           |
| [RandomGridShuffle](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomGridShuffle)             | ✓     | ✓     |        |           |
| [RandomRotate90](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomRotate90)                   | ✓     | ✓     | ✓      | ✓         |
| [RandomScale](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomScale)                         | ✓     | ✓     | ✓      | ✓         |
| [RandomSizedBBoxSafeCrop](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomSizedBBoxSafeCrop) | ✓     | ✓     | ✓      |           |
| [RandomSizedCrop](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomSizedCrop)                 | ✓     | ✓     | ✓      | ✓         |
| [Resize](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Resize)                                   | ✓     | ✓     | ✓      |           |
| [Rotate](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Rotate)                                   | ✓     | ✓     | ✓      | ✓         |
| [ShiftScaleRotate](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.ShiftScaleRotate)               | ✓     | ✓     | ✓      | ✓         |
| [SmallestMaxSize](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.SmallestMaxSize)                 | ✓     | ✓     | ✓      |           |
| [Transpose](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Transpose)                             | ✓     | ✓     | ✓      |           |
| [VerticalFlip](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.VerticalFlip)                       | ✓     | ✓     | ✓      | ✓         |

## Migrating from torchvision to albumentations

Migrating from torchvision to albumentations is simple - you just need to change a few lines of code.
Albumentations has equivalents for common torchvision transforms as well as plenty of transforms that are not presented in torchvision.
[`migrating_from_torchvision_to_albumentations.ipynb`](https://github.com/albu/albumentations/blob/master/notebooks/migrating_from_torchvision_to_albumentations.ipynb) shows how one can migrate code from torchvision to albumentations.


## Benchmarking results
To run the benchmark yourself follow the instructions in [benchmark/README.md](https://github.com/albu/albumentations/blob/master/benchmark/README.md)

Results for running the benchmark on first 2000 images from the ImageNet validation set using an Intel Xeon Gold 6140 CPU.
The table shows how many images per second can be processed on a single core, higher is better.

|                    | albumentations <br><small>0.3.0</small> | imgaug <br><small>0.2.9</small> | torchvision (Pillow backend) <br><small>0.3.0</small> | torchvision (Pillow-SIMD backend) <br><small>0.3.0</small> | Keras <br><small>2.2.4</small> | Augmentor <br><small>0.2.3</small> | <br><small>solt 0.1.6</small> |
|--------------------|:--------------------:|:------------:|:----------------------------------:|:---------------------------------------:|:-----------:|:---------------:|:----------:|
| RandomCrop64       |        **271641**    |     3373     |                26538               |                  83251                  |      -      |      22535      |    21383   |
| PadToSize512       |         **2818**     |       -      |                 414                |                   422                   |      -      |        -        |    2539    |
| Resize512          |         **2168**     |      696     |                 296                |                   1046                  |      -      |       297       |    1907    |
| HorizontalFlip     |          991         |      162     |                **4881**            |                   4595                  |     167     |       4595      |     166    |
| VerticalFlip       |         **4244**     |     2278     |                3066                |                   3598                  |     4162    |       2985      |    3486    |
| Rotate             |         **702**      |      475     |                 82                 |                   110                   |      7      |        37       |     239    |
| ShiftScaleRotate   |         **1761**     |      761     |                 81                 |                   105                   |      9      |        -        |       -    |
| Brightness         |          652         |     1520     |                 316                |                   408                   |     149     |       308       |    **2506**|
| Contrast           |         1059         |     1539     |                 231                |                   316                   |      -      |       241       |    **2524**|
| BrightnessContrast |          646         |      861     |                 128                |                   174                   |      -      |       130       |    **1305**|
| ShiftHSV           |          168         |    **208**   |                 32                 |                    42                   |      -      |        -        |     107    |
| ShiftRGB           |          555         |   **1521**   |                  -                 |                    -                    |     477     |        -        |      -     |
| Gamma              |          483         |       -      |                 801                |                   **858**               |      -      |        -        |     447    |
| Grayscale          |         **12097**    |      88      |                 728                |                   915                   |      -      |       2182      |    9140    |

Python and library versions: Python 3.7.3 | Anaconda, numpy 1.16.4, pillow 6.0.0, pillow-simd 5.3.0.post1, opencv-python 4.1.0.25, scikit-image 0.15.0, scipy 1.3.0.


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

### Adding new transforms
If you are contributing a new transformation, make sure to update ["Pixel-level transforms"](#pixel-level-transforms) or/and ["Spatial-level transforms"](#spatial-level-transforms) sections of this file (`README.md`). To do this, simply run (with python3 only):
```
python3 tools/make_transforms_docs.py make
```
and copy/paste the results in the corresponding sections. To validate your modifications, you
can run:
```
python3 tools/make_transforms_docs.py check README.md
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
