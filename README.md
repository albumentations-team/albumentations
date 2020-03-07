# Albumentations
[![PyPI version](https://badge.fury.io/py/albumentations.svg)](https://badge.fury.io/py/albumentations)
![CI](https://github.com/albumentations-team/albumentations/workflows/CI/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/albumentations/badge/?version=latest)](https://albumentations.readthedocs.io/en/latest/?badge=latest)

* The library works with images in `HWC` format.
* The library is faster than other libraries on most of the transformations.
* Based on numpy, OpenCV, imgaug picking the best from each of them.
* Simple, flexible API that allows the library to be used in any computer vision pipeline.
* Large, diverse set of transformations.
* Easy to extend the library to wrap around other libraries.
* Easy to extend to other tasks.
* Supports transformations on images, masks, key points and bounding boxes.
* Supports python 3.5-3.7
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
- [Competitions won with the library](#competitions-won-with-the-library)
- [Industry users](#used-by)

## How to use

**All in one showcase notebook** - [`showcase.ipynb`](https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/showcase.ipynb)

**Classification** - [`example.ipynb`](https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/example.ipynb)

**Object detection** - [`example_bboxes.ipynb`](https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/example_bboxes.ipynb)

**Non-8-bit images** - [`example_16_bit_tiff.ipynb`](https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/example_16_bit_tiff.ipynb)

**Image segmentation** [`example_kaggle_salt.ipynb`](https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/example_kaggle_salt.ipynb)

**Keypoints** [`example_keypoints.ipynb`](https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/example_keypoints.ipynb)

**Custom targets** [`example_multi_target.ipynb`](https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/example_multi_target.ipynb)

**Weather transforms** [`example_weather_transforms.ipynb`](https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/example_weather_transforms.ipynb)

**Serialization** [`serialization.ipynb`](https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/serialization.ipynb)

**Replay/Deterministic mode** [`replay.ipynb`](https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/replay.ipynb)

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

[Mikhail Druzhinin](https://www.linkedin.com/in/mikhail-druzhinin-548229100/)

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
- [Downscale](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Downscale)
- [Equalize](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Equalize)
- [FancyPCA](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.FancyPCA)
- [FromFloat](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.FromFloat)
- [GaussNoise](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.GaussNoise)
- [GaussianBlur](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.GaussianBlur)
- [HueSaturationValue](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.HueSaturationValue)
- [IAAAdditiveGaussianNoise](https://albumentations.readthedocs.io/en/latest/api/imgaug.html#albumentations.imgaug.transforms.IAAAdditiveGaussianNoise)
- [IAAEmboss](https://albumentations.readthedocs.io/en/latest/api/imgaug.html#albumentations.imgaug.transforms.IAAEmboss)
- [IAASharpen](https://albumentations.readthedocs.io/en/latest/api/imgaug.html#albumentations.imgaug.transforms.IAASharpen)
- [IAASuperpixels](https://albumentations.readthedocs.io/en/latest/api/imgaug.html#albumentations.imgaug.transforms.IAASuperpixels)
- [ISONoise](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.ISONoise)
- [ImageCompression](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.ImageCompression)
- [InvertImg](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.InvertImg)
- [JpegCompression](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.JpegCompression)
- [MedianBlur](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.MedianBlur)
- [MotionBlur](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.MotionBlur)
- [MultiplicativeNoise](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.MultiplicativeNoise)
- [Normalize](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Normalize)
- [Posterize](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Posterize)
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
- [GlassBlur](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.GlassBlur)
- [Solarize](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Solarize)
- [ToFloat](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.ToFloat)
- [ToGray](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.ToGray)
- [ToSepia](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.ToSepia)

## Spatial-level transforms
Spatial-level transforms will simultaneously change both an input image as well as additional targets such as masks, bounding boxes, and keypoints. The following table shows which additional targets are supported by each transform.

| Transform                                                                                                                                                           | Image | Masks | BBoxes | Keypoints |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---: | :---: | :----: | :-------: |
| [CenterCrop](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.CenterCrop)                             | ✓     | ✓     | ✓      | ✓         |
| [Crop](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Crop)                                         | ✓     | ✓     | ✓      | ✓         |
| [CropNonEmptyMaskIfExists](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.CropNonEmptyMaskIfExists) | ✓     | ✓     | ✓      | ✓         |
| [ElasticTransform](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.ElasticTransform)                 | ✓     | ✓     |        |           |
| [Flip](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Flip)                                         | ✓     | ✓     | ✓      | ✓         |
| [GridDistortion](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.GridDistortion)                     | ✓     | ✓     |        |           |
| [GridDropout](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.GridDropout)                           | ✓     | ✓     |        |           |
| [HorizontalFlip](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.HorizontalFlip)                     | ✓     | ✓     | ✓      | ✓         |
| [IAAAffine](https://albumentations.readthedocs.io/en/latest/api/imgaug.html#albumentations.imgaug.transforms.IAAAffine)                                             | ✓     | ✓     | ✓      | ✓         |
| [IAACropAndPad](https://albumentations.readthedocs.io/en/latest/api/imgaug.html#albumentations.imgaug.transforms.IAACropAndPad)                                     | ✓     | ✓     | ✓      | ✓         |
| [IAAFliplr](https://albumentations.readthedocs.io/en/latest/api/imgaug.html#albumentations.imgaug.transforms.IAAFliplr)                                             | ✓     | ✓     | ✓      | ✓         |
| [IAAFlipud](https://albumentations.readthedocs.io/en/latest/api/imgaug.html#albumentations.imgaug.transforms.IAAFlipud)                                             | ✓     | ✓     | ✓      | ✓         |
| [IAAPerspective](https://albumentations.readthedocs.io/en/latest/api/imgaug.html#albumentations.imgaug.transforms.IAAPerspective)                                   | ✓     | ✓     | ✓      | ✓         |
| [IAAPiecewiseAffine](https://albumentations.readthedocs.io/en/latest/api/imgaug.html#albumentations.imgaug.transforms.IAAPiecewiseAffine)                           | ✓     | ✓     | ✓      | ✓         |
| [Lambda](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Lambda)                                     | ✓     | ✓     | ✓      | ✓         |
| [LongestMaxSize](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.LongestMaxSize)                     | ✓     | ✓     | ✓      | ✓         |
| [MaskDropout](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.MaskDropout)                           | ✓     | ✓     |        |           |
| NoOp                                                                                                                                                                | ✓     | ✓     | ✓      | ✓         |
| [OpticalDistortion](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.OpticalDistortion)               | ✓     | ✓     |        |           |
| [PadIfNeeded](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.PadIfNeeded)                           | ✓     | ✓     | ✓      | ✓         |
| [RandomCrop](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomCrop)                             | ✓     | ✓     | ✓      | ✓         |
| [RandomCropNearBBox](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomCropNearBBox)             | ✓     | ✓     | ✓      | ✓         |
| [RandomGridShuffle](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomGridShuffle)               | ✓     | ✓     |        |           |
| [RandomResizedCrop](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomResizedCrop)               | ✓     | ✓     | ✓      | ✓         |
| [RandomRotate90](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomRotate90)                     | ✓     | ✓     | ✓      | ✓         |
| [RandomScale](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomScale)                           | ✓     | ✓     | ✓      | ✓         |
| [RandomSizedBBoxSafeCrop](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomSizedBBoxSafeCrop)   | ✓     | ✓     | ✓      |           |
| [RandomSizedCrop](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomSizedCrop)                   | ✓     | ✓     | ✓      | ✓         |
| [Resize](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Resize)                                     | ✓     | ✓     | ✓      | ✓         |
| [Rotate](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Rotate)                                     | ✓     | ✓     | ✓      | ✓         |
| [ShiftScaleRotate](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.ShiftScaleRotate)                 | ✓     | ✓     | ✓      | ✓         |
| [SmallestMaxSize](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.SmallestMaxSize)                   | ✓     | ✓     | ✓      | ✓         |
| [Transpose](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Transpose)                               | ✓     | ✓     | ✓      | ✓         |
| [VerticalFlip](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.VerticalFlip)                         | ✓     | ✓     | ✓      | ✓         |

## Migrating from torchvision to albumentations

Migrating from torchvision to albumentations is simple - you just need to change a few lines of code.
Albumentations has equivalents for common torchvision transforms as well as plenty of transforms that are not presented in torchvision.
[`migrating_from_torchvision_to_albumentations.ipynb`](https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/migrating_from_torchvision_to_albumentations.ipynb) shows how one can migrate code from torchvision to albumentations.


## Benchmarking results
To run the benchmark yourself follow the instructions in [benchmark/README.md](https://github.com/albu/albumentations/blob/master/benchmark/README.md)

Results for running the benchmark on first 2000 images from the ImageNet validation set using an Intel Xeon Platinum 8168 CPU.
All outputs are converted to a contiguous NumPy array with the np.uint8 data type.
The table shows how many images per second can be processed on a single core, higher is better.

|                      |albumentations<br><small>0.4.5</small>|imgaug<br><small>0.4.0</small>|torchvision (Pillow-SIMD backend)<br><small>0.5.0</small>|keras<br><small>2.3.1</small>|augmentor<br><small>0.2.8</small>|solt<br><small>0.1.8</small>|
|----------------------|:------------------------------------:|:----------------------------:|:-------------------------------------------------------:|:---------------------------:|:-------------------------------:|:--------------------------:|
|HorizontalFlip        |               **3066**               |             1544             |                          1652                           |             874             |              1658               |            853             |
|VerticalFlip          |               **4159**               |             2014             |                          1427                           |            4147             |              1448               |            3788            |
|Rotate                |               **417**                |             327              |                           160                           |             29              |               60                |            113             |
|ShiftScaleRotate      |               **703**                |             471              |                           144                           |             30              |                -                |             -              |
|Brightness            |               **2210**               |             997              |                           397                           |             210             |               396               |            2058            |
|Contrast              |               **2208**               |             1023             |                           330                           |              -              |               331               |            2059            |
|BrightnessContrast    |               **2199**               |             582              |                           190                           |              -              |               190               |            1051            |
|ShiftRGB              |               **2215**               |             998              |                            -                            |             378             |                -                |             -              |
|ShiftHSV              |               **381**                |             241              |                           59                            |              -              |                -                |            128             |
|Gamma                 |               **2340**               |              -               |                           686                           |              -              |                -                |            951             |
|Grayscale             |               **4961**               |             372              |                           735                           |              -              |              1423               |            4286            |
|RandomCrop64          |              **157376**              |             2560             |                          41448                          |              -              |              36036              |           35454            |
|PadToSize512          |               **2833**               |              -               |                           478                           |              -              |                -                |            2629            |
|Resize512             |               **952**                |             595              |                           885                           |              -              |               873               |            881             |
|RandomSizedCrop_64_512|               **3128**               |             881              |                          1295                           |              -              |              1254               |            2678            |
|Equalize              |               **760**                |             399              |                            -                            |              -              |               666               |             -              |
|Multiply              |               **2184**               |             1059             |                            -                            |              -              |                -                |             -              |
|MultiplyElementwise   |                 124                  |           **197**            |                            -                            |              -              |                -                |             -              |

Python and library versions: Python 3.7.5 (default, Oct 19 2019, 00:03:48) [GCC 8.3.0], numpy 1.18.1, pillow-simd 7.0.0.post3, opencv-python 4.2.0.32, scikit-image 0.16.2, scipy 1.4.1.

## Contributing

To create a pull request to the repository follow the documentation at [docs/contributing.rst](docs/contributing.rst)

### Adding new transforms
If you are contributing a new transformation, make sure to update ["Pixel-level transforms"](#pixel-level-transforms) or/and ["Spatial-level transforms"](#spatial-level-transforms) sections of this file (`README.md`). To do this, simply run (with python3 only):
```
python3 tools/make_transforms_docs.py make
```
and copy/paste the results into the corresponding sections. To validate your modifications, you
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

## Competitions won with the library
Albumentations are widely used in Computer Vision Competitions at Kaggle an other platforms.

You can find their names and links to the solutions [here](docs/hall_of_fame.rst).

## Used by
<a href="https://www.lyft.com/" target="_blank"><img src="https://habrastorage.org/webt/ce/bs/sa/cebssajf_5asst5yshmyykqjhcg.png" width="100"/></a>
<a href="https://www.x5.ru/en" target="_blank"><img src="https://habrastorage.org/webt/9y/dv/f1/9ydvf1fbxotkl6nyhydrn9v8cqw.png" width="100"/></a>
<a href="https://imedhub.org/" target="_blank"><img src="https://habrastorage.org/webt/eq/8x/m-/eq8xm-fjfx_uqkka4_ekxsdwtiq.png" width="100"/></a>
<a href="https://recursionpharma.com" target="_blank"><img src="https://pbs.twimg.com/profile_images/925897897165639683/jI8YvBfC_400x400.jpg" width="100"/></a>
<a href="https://www.everypixel.com/" target="_blank"><img src="https://www.everypixel.com/i/logo_sq.png" width="100"/></a>
<a href="https://neuromation.io/" target="_blank"><img src="https://habrastorage.org/webt/yd/_4/xa/yd_4xauvggn1tuz5xgrtkif6lya.png" width="100"/></a>

## Comments
In some systems, in the multiple GPU regime PyTorch may deadlock the DataLoader if OpenCV was compiled with OpenCL optimizations. Adding the following two lines before the library import may help. For more details [https://github.com/pytorch/pytorch/issues/1355](https://github.com/pytorch/pytorch/issues/1355)

```python
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
```

# Citing

If you find this library useful for your research, please consider citing [Albumentations: Fast and Flexible Image Augmentations](https://www.mdpi.com/2078-2489/11/2/125):

```
@Article{info11020125,
    AUTHOR = {Buslaev, Alexander and Iglovikov, Vladimir I. and Khvedchenya, Eugene and Parinov, Alex and Druzhinin, Mikhail and Kalinin, Alexandr A.},
    TITLE = {Albumentations: Fast and Flexible Image Augmentations},
    JOURNAL = {Information},
    VOLUME = {11},
    YEAR = {2020},
    NUMBER = {2},
    ARTICLE-NUMBER = {125},
    URL = {https://www.mdpi.com/2078-2489/11/2/125},
    ISSN = {2078-2489},
    DOI = {10.3390/info11020125}
}
```

You can find the full list of papers that cite Albumentations [here](docs/citations.rst).
