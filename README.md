# Albumentations
[![PyPI version](https://badge.fury.io/py/albumentations.svg)](https://badge.fury.io/py/albumentations)
![CI](https://github.com/albumentations-team/albumentations/workflows/CI/badge.svg)

Albumentations is a Python library for image augmentation. Image augmentation is used in deep learning and computer vision tasks to increase the quality of trained models. The purpose of image augmentation is to create new training samples from the existing data.

Here is an example of how you can apply some augmentations from Albumentations to create new images from the original one:
![parrot](https://habrastorage.org/webt/bd/ne/rv/bdnerv5ctkudmsaznhw4crsdfiw.jpeg)

## Why Albumentations
- Albumentations **[supports all common computer vision tasks](#i-want-to-use-albumentations-for-the-specific-task-such-as-classification-or-segmentation)** such as classification, semantic segmentation, instance segmentation, object detection, and pose estimation.
- The library provides **[a simple unified API](#a-simple-example)** to work with all data types: images (RBG-images, grayscale images, multispectral images), segmentation masks, bounding boxes, and keypoints.
- The library contains **[more than 70 different augmentations](#list-of-augmentations)** to generate new training samples from the existing data.
- Albumentations is [**fast**](#benchmarking-results). We benchmark each new release to ensure that augmentations provide maximum speed.
- It **[works with popular deep learning frameworks](#i-want-to-know-how-to-use-albumentations-with-deep-learning-frameworks)** such as PyTorch and TensorFlow 2. By the way, Albumentations is a part of the [PyTorch ecosystem](https://pytorch.org/ecosystem/).
- [**Written by experts**](#authors). The authors have experience both working on production computer vision systems and participating in competitive machine learning. Many core team members are Kaggle Masters and Grandmasters.
- The library is [**widely used**](#who-is-using-albumentations) in industry, deep learning research, machine learning competitions, and open source projects.

## Table of contents
- [Authors](#authors)
- [Installation](#installation)
- [Documentation](#documentation)
- [A simple example](#a-simple-example)
- [Getting started](#getting-started)
  - [I am new to image augmentation](#i-am-new-to-image-augmentation)
  - [I want to use Albumentations for the specific task such as classification or segmentation](#i-want-to-use-albumentations-for-the-specific-task-such-as-classification-or-segmentation)
  - [I want to know how to use Albumentations with deep learning frameworks](#i-want-to-know-how-to-use-albumentations-with-deep-learning-frameworks)
  - [I want to explore augmentations and see Albumentations in action](#i-want-to-explore-augmentations-and-see-albumentations-in-action)
- [Who is using Albumentations](#who-is-using-albumentations)
- [List of augmentations](#list-of-augmentations)
  - [Pixel-level transforms](#pixel-level-transforms)
  - [Spatial-level transforms](#spatial-level-transforms)
- [A few more examples of augmentations](#a-few-more-examples-of-augmentations)
- [Benchmarking results](#benchmarking-results)
- [Contributing](#contributing)
- [Comments](#comments)
- [Citing](#citing)

## Authors
[**Alexander Buslaev** — Computer Vision Engineer at Mapbox](https://www.linkedin.com/in/al-buslaev/) | [Kaggle Master](https://www.kaggle.com/albuslaev)

[**Alex Parinov** — Computer Vision Architect at X5 Retail Group](https://www.linkedin.com/in/alex-parinov/) | [Kaggle Master](https://www.kaggle.com/creafz)

[**Vladimir I. Iglovikov** — Senior Computer Vision Engineer at Lyft Level5](https://www.linkedin.com/in/iglovikov/) | [Kaggle Grandmaster](https://www.kaggle.com/iglovikov)

[**Evegene Khvedchenya** — AI/ML Advisor and Independent researcher](https://www.linkedin.com/in/cvtalks/) | [Kaggle Master](https://www.kaggle.com/bloodaxe)

[**Mikhail Druzhinin** — Computer Vision Engineer at Simicon](https://www.linkedin.com/in/mikhail-druzhinin-548229100/) | [Kaggle Expert](https://www.kaggle.com/dipetm)


## Installation
Albumentations requires Python 3.6 or higher. To install the latest version from PyPI:

```
pip install -U albumentations
```

Other installation options are described in the [documentation](https://albumentations.ai/docs/getting_started/installation/)

## Documentation
The full documentation is available at **[https://albumentations.ai/docs/](https://albumentations.ai/docs/)**.

## A simple example
```python
import albumentations as A
import cv2

# Declare an augmentation pipeline
transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

# Read an image with OpenCV and convert it to the RGB colorspace
image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Augment an image
transformed = transform(image=image)
transformed_image = transformed["image"]
```

## Getting started

### I am new to image augmentation
Please start with the [introduction articles](https://albumentations.ai/docs/#introduction-to-image-augmentation) about why image augmentation is important and how it helps to build better models.

### I want to use Albumentations for the specific task such as classification or segmentation
If you want to use Albumentations for a specific task such as classification, segmentation, or object detection, refer to the [set of articles](https://albumentations.ai/docs/#getting-started-with-albumentations) that has an in-depth description of this task. We also have a [list of examples](https://albumentations.ai/docs/examples/) on applying Albumentations for different use cases.

### I want to know how to use Albumentations with deep learning frameworks
We have [examples of using Albumentations](https://albumentations.ai/docs/#examples-of-how-to-use-albumentations-with-different-deep-learning-frameworks) along with PyTorch and TensorFlow 2.

### I want to explore augmentations and see Albumentations in action
Check the [online demo of the library](https://albumentations-demo.herokuapp.com/). With it, you can apply augmentations to different images and see the result. Also, we have a [list of all available augmentations and their targets](#list-of-augmentations).

## Who is using Albumentations
<a href="https://www.lyft.com/" target="_blank"><img src="https://habrastorage.org/webt/ce/bs/sa/cebssajf_5asst5yshmyykqjhcg.png" width="100"/></a>
<a href="https://www.x5.ru/en" target="_blank"><img src="https://habrastorage.org/webt/9y/dv/f1/9ydvf1fbxotkl6nyhydrn9v8cqw.png" width="100"/></a>
<a href="https://imedhub.org/" target="_blank"><img src="https://habrastorage.org/webt/eq/8x/m-/eq8xm-fjfx_uqkka4_ekxsdwtiq.png" width="100"/></a>
<a href="https://recursionpharma.com" target="_blank"><img src="https://pbs.twimg.com/profile_images/925897897165639683/jI8YvBfC_400x400.jpg" width="100"/></a>
<a href="https://www.everypixel.com/" target="_blank"><img src="https://www.everypixel.com/i/logo_sq.png" width="100"/></a>
<a href="https://neuromation.io/" target="_blank"><img src="https://habrastorage.org/webt/yd/_4/xa/yd_4xauvggn1tuz5xgrtkif6lya.png" width="100"/></a>

#### See also:
- [A list of papers that cite Albumentations](https://albumentations.ai/whos_using#research).
- [A list of teams that were using Albumentations and took high places in machine learning competitions](https://albumentations.ai/whos_using#competitions).
- [Open source projects that use Albumentations](https://albumentations.ai/whos_using#open-source).

## List of augmentations

### Pixel-level transforms
Pixel-level transforms will change just an input image and will leave any additional targets such as masks, bounding boxes, and keypoints unchanged. The list of pixel-level transforms:

- [Blur](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Blur)
- [CLAHE](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.CLAHE)
- [ChannelDropout](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ChannelDropout)
- [ChannelShuffle](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ChannelShuffle)
- [ColorJitter](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ColorJitter)
- [Downscale](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Downscale)
- [Equalize](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Equalize)
- [FDA](https://albumentations.ai/docs/api_reference/augmentations/domain_adaptation/#albumentations.augmentations.domain_adaptation.FDA)
- [FancyPCA](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.FancyPCA)
- [FromFloat](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.FromFloat)
- [GaussNoise](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.GaussNoise)
- [GaussianBlur](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.GaussianBlur)
- [GlassBlur](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.GlassBlur)
- [HistogramMatching](https://albumentations.ai/docs/api_reference/augmentations/domain_adaptation/#albumentations.augmentations.domain_adaptation.HistogramMatching)
- [HueSaturationValue](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.HueSaturationValue)
- [IAAAdditiveGaussianNoise](https://albumentations.ai/docs/api_reference/imgaug/transforms/#albumentations.imgaug.transforms.IAAAdditiveGaussianNoise)
- [IAAEmboss](https://albumentations.ai/docs/api_reference/imgaug/transforms/#albumentations.imgaug.transforms.IAAEmboss)
- [IAASharpen](https://albumentations.ai/docs/api_reference/imgaug/transforms/#albumentations.imgaug.transforms.IAASharpen)
- [IAASuperpixels](https://albumentations.ai/docs/api_reference/imgaug/transforms/#albumentations.imgaug.transforms.IAASuperpixels)
- [ISONoise](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ISONoise)
- [ImageCompression](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ImageCompression)
- [InvertImg](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.InvertImg)
- [MedianBlur](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.MedianBlur)
- [MotionBlur](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.MotionBlur)
- [MultiplicativeNoise](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.MultiplicativeNoise)
- [Normalize](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Normalize)
- [Posterize](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Posterize)
- [RGBShift](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RGBShift)
- [RandomBrightnessContrast](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomBrightnessContrast)
- [RandomFog](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomFog)
- [RandomGamma](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomGamma)
- [RandomRain](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomRain)
- [RandomShadow](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomShadow)
- [RandomSnow](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomSnow)
- [RandomSunFlare](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomSunFlare)
- [Solarize](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Solarize)
- [ToFloat](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ToFloat)
- [ToGray](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ToGray)
- [ToSepia](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ToSepia)

### Spatial-level transforms
Spatial-level transforms will simultaneously change both an input image as well as additional targets such as masks, bounding boxes, and keypoints. The following table shows which additional targets are supported by each transform.

| Transform                                                                                                                                                           | Image | Masks | BBoxes | Keypoints |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---: | :---: | :----: | :-------: |
| [CenterCrop](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.CenterCrop)                             | ✓     | ✓     | ✓      | ✓         |
| [CoarseDropout](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.CoarseDropout)                       | ✓     | ✓     |        |           |
| [Crop](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Crop)                                         | ✓     | ✓     | ✓      | ✓         |
| [CropNonEmptyMaskIfExists](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.CropNonEmptyMaskIfExists) | ✓     | ✓     | ✓      | ✓         |
| [ElasticTransform](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ElasticTransform)                 | ✓     | ✓     |        |           |
| [Flip](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Flip)                                         | ✓     | ✓     | ✓      | ✓         |
| [GridDistortion](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.GridDistortion)                     | ✓     | ✓     |        |           |
| [GridDropout](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.GridDropout)                           | ✓     | ✓     |        |           |
| [HorizontalFlip](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.HorizontalFlip)                     | ✓     | ✓     | ✓      | ✓         |
| [IAAAffine](https://albumentations.ai/docs/api_reference/imgaug/transforms/#albumentations.imgaug.transforms.IAAAffine)                                             | ✓     | ✓     | ✓      | ✓         |
| [IAACropAndPad](https://albumentations.ai/docs/api_reference/imgaug/transforms/#albumentations.imgaug.transforms.IAACropAndPad)                                     | ✓     | ✓     | ✓      | ✓         |
| [IAAFliplr](https://albumentations.ai/docs/api_reference/imgaug/transforms/#albumentations.imgaug.transforms.IAAFliplr)                                             | ✓     | ✓     | ✓      | ✓         |
| [IAAFlipud](https://albumentations.ai/docs/api_reference/imgaug/transforms/#albumentations.imgaug.transforms.IAAFlipud)                                             | ✓     | ✓     | ✓      | ✓         |
| [IAAPerspective](https://albumentations.ai/docs/api_reference/imgaug/transforms/#albumentations.imgaug.transforms.IAAPerspective)                                   | ✓     | ✓     | ✓      | ✓         |
| [IAAPiecewiseAffine](https://albumentations.ai/docs/api_reference/imgaug/transforms/#albumentations.imgaug.transforms.IAAPiecewiseAffine)                           | ✓     | ✓     | ✓      | ✓         |
| [Lambda](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Lambda)                                     | ✓     | ✓     | ✓      | ✓         |
| [LongestMaxSize](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.LongestMaxSize)                     | ✓     | ✓     | ✓      | ✓         |
| [MaskDropout](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.MaskDropout)                           | ✓     | ✓     |        |           |
| [NoOp](https://albumentations.ai/docs/api_reference/core/transforms_interface/#albumentations.core.transforms_interface.NoOp)                                       | ✓     | ✓     | ✓      | ✓         |
| [OpticalDistortion](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.OpticalDistortion)               | ✓     | ✓     |        |           |
| [PadIfNeeded](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.PadIfNeeded)                           | ✓     | ✓     | ✓      | ✓         |
| [RandomCrop](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomCrop)                             | ✓     | ✓     | ✓      | ✓         |
| [RandomCropNearBBox](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomCropNearBBox)             | ✓     | ✓     | ✓      | ✓         |
| [RandomGridShuffle](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomGridShuffle)               | ✓     | ✓     |        |           |
| [RandomResizedCrop](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomResizedCrop)               | ✓     | ✓     | ✓      | ✓         |
| [RandomRotate90](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomRotate90)                     | ✓     | ✓     | ✓      | ✓         |
| [RandomScale](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomScale)                           | ✓     | ✓     | ✓      | ✓         |
| [RandomSizedBBoxSafeCrop](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomSizedBBoxSafeCrop)   | ✓     | ✓     | ✓      |           |
| [RandomSizedCrop](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomSizedCrop)                   | ✓     | ✓     | ✓      | ✓         |
| [Resize](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Resize)                                     | ✓     | ✓     | ✓      | ✓         |
| [Rotate](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Rotate)                                     | ✓     | ✓     | ✓      | ✓         |
| [ShiftScaleRotate](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ShiftScaleRotate)                 | ✓     | ✓     | ✓      | ✓         |
| [SmallestMaxSize](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.SmallestMaxSize)                   | ✓     | ✓     | ✓      | ✓         |
| [Transpose](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Transpose)                               | ✓     | ✓     | ✓      | ✓         |
| [VerticalFlip](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.VerticalFlip)                         | ✓     | ✓     | ✓      | ✓         |



## A few more examples of augmentations
### Semantic segmentation on the Inria dataset

![inria](https://habrastorage.org/webt/su/wa/np/suwanpeo6ww7wpwtobtrzd_cg20.jpeg)

### Medical imaging
![medical](https://habrastorage.org/webt/1i/fi/wz/1ifiwzy0lxetc4nwjvss-71nkw0.jpeg)

### Object detection and semantic segmentation on the Mapillary Vistas dataset
![vistas](https://habrastorage.org/webt/rz/-h/3j/rz-h3jalbxic8o_fhucxysts4tc.jpeg)

### Keypoints augmentation
<img src="https://habrastorage.org/webt/e-/6k/z-/e-6kz-fugp2heak3jzns3bc-r8o.jpeg" width=100%>


## Benchmarking results
To run the benchmark yourself, follow the instructions in [benchmark/README.md](https://github.com/albumentations-team/albumentations/blob/master/benchmark/README.md)

Results for running the benchmark on the first 2000 images from the ImageNet validation set using an Intel Xeon Gold 6140 CPU.
All outputs are converted to a contiguous NumPy array with the np.uint8 data type.
The table shows how many images per second can be processed on a single core; higher is better.

|                      |albumentations<br><small>0.5.0</small>|imgaug<br><small>0.4.0</small>|torchvision (Pillow-SIMD backend)<br><small>0.7.0</small>|keras<br><small>2.4.3</small>|augmentor<br><small>0.2.8</small>|solt<br><small>0.1.9</small>|
|----------------------|:------------------------------------:|:----------------------------:|:-------------------------------------------------------:|:---------------------------:|:-------------------------------:|:--------------------------:|
|HorizontalFlip        |               **9909**               |             2821             |                          2267                           |             873             |              2301               |            6223            |
|VerticalFlip          |               **4374**               |             2218             |                          1952                           |            4339             |              1968               |            3562            |
|Rotate                |               **371**                |             296              |                           163                           |             27              |               60                |            345             |
|ShiftScaleRotate      |               **635**                |             437              |                           147                           |             28              |                -                |             -              |
|Brightness            |               **2751**               |             1178             |                           419                           |             229             |               418               |            2300            |
|Contrast              |               **2756**               |             1213             |                           352                           |              -              |               348               |            2305            |
|BrightnessContrast    |               **2738**               |             699              |                           195                           |              -              |               193               |            1179            |
|ShiftRGB              |               **2757**               |             1176             |                            -                            |             348             |                -                |             -              |
|ShiftHSV              |               **597**                |             284              |                           58                            |              -              |                -                |            137             |
|Gamma                 |               **2844**               |              -               |                           382                           |              -              |                -                |            946             |
|Grayscale             |               **5159**               |             428              |                           709                           |              -              |              1064               |            1273            |
|RandomCrop64          |              **175886**              |             3018             |                          52103                          |              -              |              41774              |           20732            |
|PadToSize512          |               **3418**               |              -               |                           574                           |              -              |                -                |            2874            |
|Resize512             |                 1003                 |             634              |                        **1036**                         |              -              |              1016               |            977             |
|RandomSizedCrop_64_512|               **3191**               |             939              |                          1594                           |              -              |              1529               |            2563            |
|Posterize             |               **2778**               |              -               |                            -                            |              -              |                -                |             -              |
|Solarize              |               **2762**               |              -               |                            -                            |              -              |                -                |             -              |
|Equalize              |                 644                  |             413              |                            -                            |              -              |             **735**             |             -              |
|Multiply              |               **2727**               |             1248             |                            -                            |              -              |                -                |             -              |
|MultiplyElementwise   |                 118                  |           **209**            |                            -                            |              -              |                -                |             -              |
|ColorJitter           |               **368**                |              78              |                           57                            |              -              |                -                |             -              |

Python and library versions: Python 3.8.6 (default, Oct 13 2020, 20:37:26) [GCC 8.3.0], numpy 1.19.2, pillow-simd 7.0.0.post3, opencv-python 4.4.0.44, scikit-image 0.17.2, scipy 1.5.2.

## Contributing

To create a pull request to the repository, follow the documentation at [https://albumentations.ai/docs/contributing/](https://albumentations.ai/docs/contributing/)


## Comments
In some systems, in the multiple GPU regime, PyTorch may deadlock the DataLoader if OpenCV was compiled with OpenCL optimizations. Adding the following two lines before the library import may help. For more details [https://github.com/pytorch/pytorch/issues/1355](https://github.com/pytorch/pytorch/issues/1355)

```python
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
```

## Citing

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
