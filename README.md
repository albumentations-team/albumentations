# Albumentations

[![PyPI version](https://badge.fury.io/py/albumentations.svg)](https://badge.fury.io/py/albumentations)
![CI](https://github.com/albumentations-team/albumentations/workflows/CI/badge.svg)
[![PyPI Downloads](https://img.shields.io/pypi/dm/albumentations.svg?label=PyPI%20downloads)](
https://pypi.org/project/albumentations/)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/albumentations.svg?label=Conda%20downloads)](
https://anaconda.org/conda-forge/albumentations)
[![Stack Overflow](https://img.shields.io/badge/stackoverflow-Ask%20questions-blue.svg)](
https://stackoverflow.com/questions/tagged/albumentations)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)

[Docs](https://albumentations.ai/docs/) | [Discord](https://discord.gg/AKPrrDYNAt) | [Twitter](https://twitter.com/albumentations) | [LinkedIn](https://www.linkedin.com/company/100504475/)

Albumentations is a Python library for image augmentation. Image augmentation is used in deep learning and computer vision tasks to increase the quality of trained models. The purpose of image augmentation is to create new training samples from the existing data.

Here is an example of how you can apply some [pixel-level](#pixel-level-transforms) augmentations from Albumentations to create new images from the original one:
![parrot](https://habrastorage.org/webt/bd/ne/rv/bdnerv5ctkudmsaznhw4crsdfiw.jpeg)

## Why Albumentations

- Albumentations **[supports all common computer vision tasks](#i-want-to-use-albumentations-for-the-specific-task-such-as-classification-or-segmentation)** such as classification, semantic segmentation, instance segmentation, object detection, and pose estimation.
- The library provides **[a simple unified API](#a-simple-example)** to work with all data types: images (RBG-images, grayscale images, multispectral images), segmentation masks, bounding boxes, and keypoints.
- The library contains **[more than 70 different augmentations](#list-of-augmentations)** to generate new training samples from the existing data.
- Albumentations is [**fast**](#benchmarking-results). We benchmark each new release to ensure that augmentations provide maximum speed.
- It **[works with popular deep learning frameworks](#i-want-to-know-how-to-use-albumentations-with-deep-learning-frameworks)** such as PyTorch and TensorFlow. By the way, Albumentations is a part of the [PyTorch ecosystem](https://pytorch.org/ecosystem/).
- [**Written by experts**](#authors). The authors have experience both working on production computer vision systems and participating in competitive machine learning. Many core team members are Kaggle Masters and Grandmasters.
- The library is [**widely used**](#who-is-using-albumentations) in industry, deep learning research, machine learning competitions, and open source projects.

## Sponsors

<a href="https://roboflow.com/" target="_blank"><img src="https://avatars.githubusercontent.com/u/53104118?s=200&v=4" width="100"/></a>

## Table of contents

- [Albumentations](#albumentations)
  - [Why Albumentations](#why-albumentations)
  - [Sponsors](#sponsors)
  - [Table of contents](#table-of-contents)
  - [Authors](#authors)
    - [Current Maintainer](#current-maintainer)
    - [Emeritus Core Team Members](#emeritus-core-team-members)
  - [Installation](#installation)
  - [Documentation](#documentation)
  - [A simple example](#a-simple-example)
  - [Getting started](#getting-started)
    - [I am new to image augmentation](#i-am-new-to-image-augmentation)
    - [I want to use Albumentations for the specific task such as classification or segmentation](#i-want-to-use-albumentations-for-the-specific-task-such-as-classification-or-segmentation)
    - [I want to know how to use Albumentations with deep learning frameworks](#i-want-to-know-how-to-use-albumentations-with-deep-learning-frameworks)
    - [I want to explore augmentations and see Albumentations in action](#i-want-to-explore-augmentations-and-see-albumentations-in-action)
  - [Who is using Albumentations](#who-is-using-albumentations)
    - [See also](#see-also)
  - [List of augmentations](#list-of-augmentations)
    - [Pixel-level transforms](#pixel-level-transforms)
    - [Spatial-level transforms](#spatial-level-transforms)
  - [A few more examples of **augmentations**](#a-few-more-examples-of-augmentations)
    - [Semantic segmentation on the Inria dataset](#semantic-segmentation-on-the-inria-dataset)
    - [Medical imaging](#medical-imaging)
    - [Object detection and semantic segmentation on the Mapillary Vistas dataset](#object-detection-and-semantic-segmentation-on-the-mapillary-vistas-dataset)
    - [Keypoints augmentation](#keypoints-augmentation)
  - [Benchmarking results](#benchmarking-results)
  - [Contributing](#contributing)
  - [Community and Support](#community-and-support)
  - [Comments](#comments)
  - [Citing](#citing)

## Authors

### Current Maintainer

[**Vladimir I. Iglovikov**](https://www.linkedin.com/in/iglovikov/) | [Kaggle Grandmaster](https://www.kaggle.com/iglovikov)

### Emeritus Core Team Members

[**Mikhail Druzhinin**](https://www.linkedin.com/in/mikhail-druzhinin-548229100/) | [Kaggle Expert](https://www.kaggle.com/dipetm)

[**Alex Parinov**](https://www.linkedin.com/in/alex-parinov/) | [Kaggle Master](https://www.kaggle.com/creafz)

[**Alexander Buslaev**](https://www.linkedin.com/in/al-buslaev/) | [Kaggle Master](https://www.kaggle.com/albuslaev)

[**Eugene Khvedchenya**](https://www.linkedin.com/in/cvtalks/) | [Kaggle Grandmaster](https://www.kaggle.com/bloodaxe)

## Installation

Albumentations requires Python 3.9 or higher. To install the latest version from PyPI:

```bash
pip install -U albumentations
```

Other installation options are described in the [documentation](https://albumentations.ai/docs/getting_started/installation/).

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

We have [examples of using Albumentations](https://albumentations.ai/docs/#examples-of-how-to-use-albumentations-with-different-deep-learning-frameworks) along with PyTorch and TensorFlow.

### I want to explore augmentations and see Albumentations in action

Check the [online demo of the library](https://albumentations-demo.herokuapp.com/). With it, you can apply augmentations to different images and see the result. Also, we have a [list of all available augmentations and their targets](#list-of-augmentations).

## Who is using Albumentations

<a href="https://www.apple.com/" target="_blank"><img src="https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/html/assets/img/industry/apple.jpeg" width="100"/></a>
<a href="https://research.google/" target="_blank"><img src="https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/html/assets/img/industry/google.png" width="100"/></a>
<a href="https://opensource.fb.com/" target="_blank"><img src="https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/html/assets/img/industry/meta_research.png" width="100"/></a>
<a href="https: //www.nvidia.com/en-us/research/" target="_blank"><img src="https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/html/assets/img/industry/nvidia_research.jpeg" width="100"/></a>
<a href="https://www.amazon.science/" target="_blank"><img src="https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/html/assets/img/industry/amazon_science.png" width="100"/></a>
<a href="https://opensource.microsoft.com/" target="_blank"><img src="https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/html/assets/img/industry/microsoft.png" width="100"/></a>
<a href="https://engineering.salesforce.com/open-source/" target="_blank"><img src="https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/html/assets/img/industry/salesforce_open_source.png" width="100"/></a>
<a href="https://stability.ai/" target="_blank"><img src="https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/html/assets/img/industry/stability.png" width="100"/></a>
<a href="https://www.ibm.com/opensource/" target="_blank"><img src="https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/html/assets/img/industry/ibm.jpeg" width="100"/></a>
<a href="https://huggingface.co/" target="_blank"><img src="https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/html/assets/img/industry/hugging_face.png" width="100"/></a>
<a href="https://www.sony.com/en/" target="_blank"><img src="https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/html/assets/img/industry/sony.png" width="100"/></a>
<a href="https://opensource.alibaba.com/" target="_blank"><img src="https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/html/assets/img/industry/alibaba.png" width="100"/></a>
<a href="https://opensource.tencent.com/" target="_blank"><img src="https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/html/assets/img/industry/tencent.png" width="100"/></a>
<a href="https://h2o.ai/" target="_blank"><img src="https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/html/assets/img/industry/h2o_ai.png" width="100"/></a>

### See also

- [A list of papers that cite Albumentations](https://scholar.google.com/citations?view_op=view_citation&citation_for_view=vkjh9X0AAAAJ:r0BpntZqJG4C).
- [A list of teams that were using Albumentations and took high places in machine learning competitions](https://albumentations.ai/whos_using#competitions).
- [Open source projects that use Albumentations](https://github.com/albumentations-team/albumentations/network/dependents?dependent_type=PACKAGE).

## List of augmentations

### Pixel-level transforms

Pixel-level transforms will change just an input image and will leave any additional targets such as masks, bounding boxes, and keypoints unchanged. The list of pixel-level transforms:

- [AdvancedBlur](https://explore.albumentations.ai/transform/AdvancedBlur)
- [Blur](https://explore.albumentations.ai/transform/Blur)
- [CLAHE](https://explore.albumentations.ai/transform/CLAHE)
- [ChannelDropout](https://explore.albumentations.ai/transform/ChannelDropout)
- [ChannelShuffle](https://explore.albumentations.ai/transform/ChannelShuffle)
- [ChromaticAberration](https://explore.albumentations.ai/transform/ChromaticAberration)
- [ColorJitter](https://explore.albumentations.ai/transform/ColorJitter)
- [Defocus](https://explore.albumentations.ai/transform/Defocus)
- [Downscale](https://explore.albumentations.ai/transform/Downscale)
- [Emboss](https://explore.albumentations.ai/transform/Emboss)
- [Equalize](https://explore.albumentations.ai/transform/Equalize)
- [FDA](https://explore.albumentations.ai/transform/FDA)
- [FancyPCA](https://explore.albumentations.ai/transform/FancyPCA)
- [FromFloat](https://explore.albumentations.ai/transform/FromFloat)
- [GaussNoise](https://explore.albumentations.ai/transform/GaussNoise)
- [GaussianBlur](https://explore.albumentations.ai/transform/GaussianBlur)
- [GlassBlur](https://explore.albumentations.ai/transform/GlassBlur)
- [HistogramMatching](https://explore.albumentations.ai/transform/HistogramMatching)
- [HueSaturationValue](https://explore.albumentations.ai/transform/HueSaturationValue)
- [ISONoise](https://explore.albumentations.ai/transform/ISONoise)
- [ImageCompression](https://explore.albumentations.ai/transform/ImageCompression)
- [InvertImg](https://explore.albumentations.ai/transform/InvertImg)
- [MedianBlur](https://explore.albumentations.ai/transform/MedianBlur)
- [MotionBlur](https://explore.albumentations.ai/transform/MotionBlur)
- [MultiplicativeNoise](https://explore.albumentations.ai/transform/MultiplicativeNoise)
- [Normalize](https://explore.albumentations.ai/transform/Normalize)
- [PixelDistributionAdaptation](https://explore.albumentations.ai/transform/PixelDistributionAdaptation)
- [PlanckianJitter](https://explore.albumentations.ai/transform/PlanckianJitter)
- [Posterize](https://explore.albumentations.ai/transform/Posterize)
- [RGBShift](https://explore.albumentations.ai/transform/RGBShift)
- [RandomBrightnessContrast](https://explore.albumentations.ai/transform/RandomBrightnessContrast)
- [RandomFog](https://explore.albumentations.ai/transform/RandomFog)
- [RandomGamma](https://explore.albumentations.ai/transform/RandomGamma)
- [RandomGravel](https://explore.albumentations.ai/transform/RandomGravel)
- [RandomRain](https://explore.albumentations.ai/transform/RandomRain)
- [RandomShadow](https://explore.albumentations.ai/transform/RandomShadow)
- [RandomSnow](https://explore.albumentations.ai/transform/RandomSnow)
- [RandomSunFlare](https://explore.albumentations.ai/transform/RandomSunFlare)
- [RandomToneCurve](https://explore.albumentations.ai/transform/RandomToneCurve)
- [RingingOvershoot](https://explore.albumentations.ai/transform/RingingOvershoot)
- [Sharpen](https://explore.albumentations.ai/transform/Sharpen)
- [Solarize](https://explore.albumentations.ai/transform/Solarize)
- [Spatter](https://explore.albumentations.ai/transform/Spatter)
- [Superpixels](https://explore.albumentations.ai/transform/Superpixels)
- [TemplateTransform](https://explore.albumentations.ai/transform/TemplateTransform)
- [TextImage](https://explore.albumentations.ai/transform/TextImage)
- [ToFloat](https://explore.albumentations.ai/transform/ToFloat)
- [ToGray](https://explore.albumentations.ai/transform/ToGray)
- [ToRGB](https://explore.albumentations.ai/transform/ToRGB)
- [ToSepia](https://explore.albumentations.ai/transform/ToSepia)
- [UnsharpMask](https://explore.albumentations.ai/transform/UnsharpMask)
- [ZoomBlur](https://explore.albumentations.ai/transform/ZoomBlur)

### Spatial-level transforms

Spatial-level transforms will simultaneously change both an input image as well as additional targets such as masks, bounding boxes, and keypoints. The following table shows which additional targets are supported by each transform.

| Transform                                                                                        | Image | Mask | BBoxes | Keypoints |
| ------------------------------------------------------------------------------------------------ | :---: | :--: | :----: | :-------: |
| [Affine](https://explore.albumentations.ai/transform/Affine)                                     | ✓     | ✓    | ✓      | ✓         |
| [BBoxSafeRandomCrop](https://explore.albumentations.ai/transform/BBoxSafeRandomCrop)             | ✓     | ✓    | ✓      | ✓         |
| [CenterCrop](https://explore.albumentations.ai/transform/CenterCrop)                             | ✓     | ✓    | ✓      | ✓         |
| [CoarseDropout](https://explore.albumentations.ai/transform/CoarseDropout)                       | ✓     | ✓    | ✓      | ✓         |
| [Crop](https://explore.albumentations.ai/transform/Crop)                                         | ✓     | ✓    | ✓      | ✓         |
| [CropAndPad](https://explore.albumentations.ai/transform/CropAndPad)                             | ✓     | ✓    | ✓      | ✓         |
| [CropNonEmptyMaskIfExists](https://explore.albumentations.ai/transform/CropNonEmptyMaskIfExists) | ✓     | ✓    | ✓      | ✓         |
| [D4](https://explore.albumentations.ai/transform/D4)                                             | ✓     | ✓    | ✓      | ✓         |
| [ElasticTransform](https://explore.albumentations.ai/transform/ElasticTransform)                 | ✓     | ✓    | ✓      | ✓         |
| [GridDistortion](https://explore.albumentations.ai/transform/GridDistortion)                     | ✓     | ✓    | ✓      | ✓         |
| [GridDropout](https://explore.albumentations.ai/transform/GridDropout)                           | ✓     | ✓    | ✓      | ✓         |
| [GridElasticDeform](https://explore.albumentations.ai/transform/GridElasticDeform)               | ✓     | ✓    | ✓      | ✓         |
| [HorizontalFlip](https://explore.albumentations.ai/transform/HorizontalFlip)                     | ✓     | ✓    | ✓      | ✓         |
| [Lambda](https://explore.albumentations.ai/transform/Lambda)                                     | ✓     | ✓    | ✓      | ✓         |
| [LongestMaxSize](https://explore.albumentations.ai/transform/LongestMaxSize)                     | ✓     | ✓    | ✓      | ✓         |
| [MaskDropout](https://explore.albumentations.ai/transform/MaskDropout)                           | ✓     | ✓    | ✓      | ✓         |
| [Morphological](https://explore.albumentations.ai/transform/Morphological)                       | ✓     | ✓    | ✓      | ✓         |
| [NoOp](https://explore.albumentations.ai/transform/NoOp)                                         | ✓     | ✓    | ✓      | ✓         |
| [OpticalDistortion](https://explore.albumentations.ai/transform/OpticalDistortion)               | ✓     | ✓    | ✓      | ✓         |
| [OverlayElements](https://explore.albumentations.ai/transform/OverlayElements)                   | ✓     | ✓    |        |           |
| [PadIfNeeded](https://explore.albumentations.ai/transform/PadIfNeeded)                           | ✓     | ✓    | ✓      | ✓         |
| [Perspective](https://explore.albumentations.ai/transform/Perspective)                           | ✓     | ✓    | ✓      | ✓         |
| [PiecewiseAffine](https://explore.albumentations.ai/transform/PiecewiseAffine)                   | ✓     | ✓    | ✓      | ✓         |
| [PixelDropout](https://explore.albumentations.ai/transform/PixelDropout)                         | ✓     | ✓    | ✓      | ✓         |
| [RandomCrop](https://explore.albumentations.ai/transform/RandomCrop)                             | ✓     | ✓    | ✓      | ✓         |
| [RandomCropFromBorders](https://explore.albumentations.ai/transform/RandomCropFromBorders)       | ✓     | ✓    | ✓      | ✓         |
| [RandomGridShuffle](https://explore.albumentations.ai/transform/RandomGridShuffle)               | ✓     | ✓    |        | ✓         |
| [RandomResizedCrop](https://explore.albumentations.ai/transform/RandomResizedCrop)               | ✓     | ✓    | ✓      | ✓         |
| [RandomRotate90](https://explore.albumentations.ai/transform/RandomRotate90)                     | ✓     | ✓    | ✓      | ✓         |
| [RandomScale](https://explore.albumentations.ai/transform/RandomScale)                           | ✓     | ✓    | ✓      | ✓         |
| [RandomSizedBBoxSafeCrop](https://explore.albumentations.ai/transform/RandomSizedBBoxSafeCrop)   | ✓     | ✓    | ✓      | ✓         |
| [RandomSizedCrop](https://explore.albumentations.ai/transform/RandomSizedCrop)                   | ✓     | ✓    | ✓      | ✓         |
| [Resize](https://explore.albumentations.ai/transform/Resize)                                     | ✓     | ✓    | ✓      | ✓         |
| [Rotate](https://explore.albumentations.ai/transform/Rotate)                                     | ✓     | ✓    | ✓      | ✓         |
| [SafeRotate](https://explore.albumentations.ai/transform/SafeRotate)                             | ✓     | ✓    | ✓      | ✓         |
| [ShiftScaleRotate](https://explore.albumentations.ai/transform/ShiftScaleRotate)                 | ✓     | ✓    | ✓      | ✓         |
| [SmallestMaxSize](https://explore.albumentations.ai/transform/SmallestMaxSize)                   | ✓     | ✓    | ✓      | ✓         |
| [Transpose](https://explore.albumentations.ai/transform/Transpose)                               | ✓     | ✓    | ✓      | ✓         |
| [VerticalFlip](https://explore.albumentations.ai/transform/VerticalFlip)                         | ✓     | ✓    | ✓      | ✓         |
| [XYMasking](https://explore.albumentations.ai/transform/XYMasking)                               | ✓     | ✓    | ✓      | ✓         |

## A few more examples of **augmentations**

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

Results for running the benchmark on the first 2000 images from the ImageNet validation set using an AMD Ryzen Threadripper 3970X CPU.
The table shows how many images per second can be processed on a single core; higher is better.

| Library | Version |
|---------|---------|
| Python | 3.10.13 (main, Sep 11 2023, 13:44:35) [GCC 11.2.0] |
| albumentations | 1.4.11 |
| imgaug | 0.4.0 |
| torchvision | 0.18.1+rocm6.0 |
| numpy | 1.26.4 |
| opencv-python-headless | 4.10.0.84 |
| scikit-image | 0.24.0 |
| scipy | 1.14.0 |
| pillow | 10.4.0 |
| kornia | 0.7.3 |
| augly | 1.0.0 |

|                 |albumentations<br><small>1.4.11</small>|torchvision<br><small>0.18.1+rocm6.0</small>|kornia<br><small>0.7.3</small>|augly<br><small>1.0.0</small>|imgaug<br><small>0.4.0</small>|
|-----------------|--------------------------------------|--------------------------------------------|------------------------------|-----------------------------|------------------------------|
|HorizontalFlip   |**8017 ± 12**                         |2436 ± 2                                    |935 ± 3                       |3575 ± 4                     |4806 ± 7                      |
|VerticalFlip     |7366 ± 7                              |2563 ± 8                                    |943 ± 1                       |4949 ± 5                     |**8159 ± 21**                 |
|Rotate           |570 ± 12                              |152 ± 2                                     |207 ± 1                       |**633 ± 2**                  |496 ± 2                       |
|Affine           |**1382 ± 31**                         |162 ± 1                                     |201 ± 1                       |-                            |682 ± 2                       |
|Equalize         |1027 ± 2                              |336 ± 2                                     |77 ± 1                        |-                            |**1183 ± 1**                  |
|RandomCrop64     |**19986 ± 57**                        |15336 ± 16                                  |811 ± 1                       |19882 ± 356                  |5410 ± 5                      |
|RandomResizedCrop|**2308 ± 7**                          |1046 ± 3                                    |187 ± 1                       |-                            |-                             |
|ShiftRGB         |1240 ± 3                              |-                                           |425 ± 2                       |-                            |**1554 ± 6**                  |
|Resize           |**2314 ± 9**                          |1272 ± 3                                    |201 ± 3                       |431 ± 1                      |1715 ± 2                      |
|RandomGamma      |**2552 ± 2**                          |232 ± 1                                     |211 ± 1                       |-                            |1794 ± 1                      |
|Grayscale        |**7313 ± 4**                          |1652 ± 2                                    |443 ± 2                       |2639 ± 2                     |1171 ± 23                     |
|ColorJitter      |**396 ± 1**                           |51 ± 1                                      |50 ± 1                        |224 ± 1                      |-                             |
|PlankianJitter   |449 ± 1                               |-                                           |**598 ± 1**                   |-                            |-                             |
|RandomPerspective|471 ± 1                               |123 ± 1                                     |114 ± 1                       |-                            |**478 ± 2**                   |
|GaussianBlur     |**2099 ± 2**                          |113 ± 2                                     |79 ± 2                        |165 ± 1                      |1244 ± 2                      |
|MedianBlur       |538 ± 1                               |-                                           |3 ± 1                         |-                            |**565 ± 1**                   |
|MotionBlur       |**2197 ± 9**                          |-                                           |102 ± 1                       |-                            |508 ± 1                       |
|Posterize        |2449 ± 1                              |**2587 ± 3**                                |339 ± 6                       |-                            |1547 ± 1                      |
|JpegCompression  |**827 ± 1**                           |-                                           |50 ± 2                        |684 ± 1                      |428 ± 4                       |
|GaussianNoise    |78 ± 1                                |-                                           |-                             |67 ± 1                       |**128 ± 1**                   |
|Elastic          |127 ± 1                               |3 ± 1                                       |1 ± 1                         |-                            |**130 ± 1**                   |
|Normalize        |**971 ± 2**                           |449 ± 1                                     |415 ± 1                       |-                            |-                             |

## Contributing

To create a pull request to the repository, follow the documentation at [CONTRIBUTING.md](CONTRIBUTING.md)

![https://github.com/albuemntations-team/albumentation/graphs/contributors](https://contrib.rocks/image?repo=albumentations-team/albumentations)

## Community and Support

- [Twitter](https://twitter.com/albumentations)
- [Discord](https://discord.gg/AKPrrDYNAt)

## Comments

In some systems, in the multiple GPU regime, PyTorch may deadlock the DataLoader if OpenCV was compiled with OpenCL optimizations. Adding the following two lines before the library import may help. For more details [https://github.com/pytorch/pytorch/issues/1355](https://github.com/pytorch/pytorch/issues/1355)

```python
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
```

## Citing

If you find this library useful for your research, please consider citing [Albumentations: Fast and Flexible Image Augmentations](https://www.mdpi.com/2078-2489/11/2/125):

```bibtex
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
