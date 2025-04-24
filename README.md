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
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20Albumentations%20Guru-006BFF)](https://gurubase.io/g/albumentations)
![PyPI - Types](https://img.shields.io/pypi/types/albumentations)

[Docs](https://albumentations.ai/docs/) | [Discord](https://discord.gg/AKPrrDYNAt) | [Twitter](https://twitter.com/albumentations) | [LinkedIn](https://www.linkedin.com/company/100504475/)

Albumentations is a Python library for image augmentation. Image augmentation is used in deep learning and computer vision tasks to increase the quality of trained models. The purpose of image augmentation is to create new training samples from the existing data.

Here is an example of how you can apply some [pixel-level](#pixel-level-transforms) augmentations from Albumentations to create new images from the original one:
![parrot](https://habrastorage.org/webt/bd/ne/rv/bdnerv5ctkudmsaznhw4crsdfiw.jpeg)

## Why Albumentations

- **Complete Computer Vision Support**: Works with [all major CV tasks](#i-want-to-use-albumentations-for-the-specific-task-such-as-classification-or-segmentation) including classification, segmentation (semantic & instance), object detection, and pose estimation.
- **Simple, Unified API**: [One consistent interface](#a-simple-example) for all data types - RGB/grayscale/multispectral images, masks, bounding boxes, and keypoints.
- **Rich Augmentation Library**: [70+ high-quality augmentations](https://albumentations.ai/docs/api_reference/augmentations/transforms/) to enhance your training data.
- **Fast**: Consistently benchmarked as the [fastest augmentation library](https://albumentations.ai/docs/benchmarking_results/#performance-comparison) also shown [below section](#performance-comparison), with optimizations for production use.
- **Deep Learning Integration**: Works with [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/), and other frameworks. Part of the [PyTorch ecosystem](https://pytorch.org/ecosystem/).
- **Created by Experts**: Built by [developers with deep experience in computer vision and machine learning competitions](#authors).

## Community-Driven Project, Supported By

Albumentations thrives on developer contributions. We appreciate our sponsors who help sustain the project's infrastructure.

| 🏆 Gold Sponsors |
|-----------------|
| Your company could be here |

| 🥈 Silver Sponsors |
|-------------------|
| <a href="https://datature.io" target="_blank"><img src="https://albumentations.ai/assets/sponsors/datature-full.png" width="100" alt="Datature"/></a> |

| 🥉 Bronze Sponsors |
|-------------------|
| <a href="https://roboflow.com" target="_blank"><img src="https://albumentations.ai/assets/sponsors/roboflow.png" width="100" alt="Roboflow"/></a> |

---

### 💝 Become a Sponsor

Your sponsorship is a way to say "thank you" to the maintainers and contributors who spend their free time building and maintaining Albumentations. Sponsors are featured on our website and README. View sponsorship tiers on [GitHub Sponsors](https://github.com/sponsors/albumentations-team)

## Table of contents

- [Albumentations](#albumentations)
  - [Why Albumentations](#why-albumentations)
  - [Community-Driven Project, Supported By](#community-driven-project-supported-by)
    - [💝 Become a Sponsor](#-become-a-sponsor)
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
  - [Benchmarking results](#benchmark-results)
    - [System Information](#system-information)
    - [Benchmark Parameters](#benchmark-parameters)
    - [Library Versions](#library-versions)
  - [Performance Comparison](#performance-comparison)
  - [Contributing](#contributing)
  - [Community](#community)
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

Please start with the [introduction articles](https://albumentations.ai/docs/#learning-path) about why image augmentation is important and how it helps to build better models.

### I want to use Albumentations for the specific task such as classification or segmentation

If you want to use Albumentations for a specific task such as classification, segmentation, or object detection, refer to the [set of articles](https://albumentations.ai/docs/#quick-start-guide) that has an in-depth description of this task. We also have a [list of examples](https://albumentations.ai/docs/examples/) on applying Albumentations for different use cases.

### I want to know how to use Albumentations with deep learning frameworks

We have [examples of using Albumentations](https://albumentations.ai/docs/#examples-of-how-to-use-albumentations-with-different-deep-learning-frameworks) along with PyTorch and TensorFlow.

### I want to explore augmentations and see Albumentations in action

Check the [online demo of the library](https://albumentations-demo.herokuapp.com/). With it, you can apply augmentations to different images and see the result. Also, we have a [list of all available augmentations and their targets](#list-of-augmentations).

## Who is using Albumentations

<a href="https://www.apple.com/" target="_blank"><img src="https://www.albumentations.ai/assets/industry/apple.jpeg" width="100"/></a>
<a href="https://research.google/" target="_blank"><img src="https://www.albumentations.ai/assets/industry/google.png" width="100"/></a>
<a href="https://opensource.fb.com/" target="_blank"><img src="https://www.albumentations.ai/assets/industry/meta_research.png" width="100"/></a>
<a href="https://www.nvidia.com/en-us/research/" target="_blank"><img src="https://www.albumentations.ai/assets/industry/nvidia_research.jpeg" width="100"/></a>
<a href="https://www.amazon.science/" target="_blank"><img src="https://www.albumentations.ai/assets/industry/amazon_science.png" width="100"/></a>
<a href="https://opensource.microsoft.com/" target="_blank"><img src="https://www.albumentations.ai/assets/industry/microsoft.png" width="100"/></a>
<a href="https://engineering.salesforce.com/open-source/" target="_blank"><img src="https://www.albumentations.ai/assets/industry/salesforce_open_source.png" width="100"/></a>
<a href="https://stability.ai/" target="_blank"><img src="https://www.albumentations.ai/assets/industry/stability.png" width="100"/></a>
<a href="https://www.ibm.com/opensource/" target="_blank"><img src="https://www.albumentations.ai/assets/industry/ibm.jpeg" width="100"/></a>
<a href="https://huggingface.co/" target="_blank"><img src="https://www.albumentations.ai/assets/industry/hugging_face.png" width="100"/></a>
<a href="https://www.sony.com/en/" target="_blank"><img src="https://www.albumentations.ai/assets/industry/sony.png" width="100"/></a>
<a href="https://opensource.alibaba.com/" target="_blank"><img src="https://www.albumentations.ai/assets/industry/alibaba.png" width="100"/></a>
<a href="https://opensource.tencent.com/" target="_blank"><img src="https://www.albumentations.ai/assets/industry/tencent.png" width="100"/></a>
<a href="https://h2o.ai/" target="_blank"><img src="https://www.albumentations.ai/assets/industry/h2o_ai.png" width="100"/></a>

### See also

- [A list of papers that cite Albumentations](https://scholar.google.com/citations?view_op=view_citation&citation_for_view=vkjh9X0AAAAJ:r0BpntZqJG4C).
- [Open source projects that use Albumentations](https://github.com/albumentations-team/albumentations/network/dependents?dependent_type=PACKAGE).

## List of augmentations

### Pixel-level transforms

Pixel-level transforms will change just an input image and will leave any additional targets such as masks, bounding boxes, and keypoints unchanged. For volumetric data (volumes and 3D masks), these transforms are applied independently to each slice along the Z-axis (depth dimension), maintaining consistency across the volume. The list of pixel-level transforms:

- [AdditiveNoise](https://explore.albumentations.ai/transform/AdditiveNoise)
- [AdvancedBlur](https://explore.albumentations.ai/transform/AdvancedBlur)
- [AutoContrast](https://explore.albumentations.ai/transform/AutoContrast)
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
- [HEStain](https://explore.albumentations.ai/transform/HEStain)
- [HistogramMatching](https://explore.albumentations.ai/transform/HistogramMatching)
- [HueSaturationValue](https://explore.albumentations.ai/transform/HueSaturationValue)
- [ISONoise](https://explore.albumentations.ai/transform/ISONoise)
- [Illumination](https://explore.albumentations.ai/transform/Illumination)
- [ImageCompression](https://explore.albumentations.ai/transform/ImageCompression)
- [InvertImg](https://explore.albumentations.ai/transform/InvertImg)
- [MedianBlur](https://explore.albumentations.ai/transform/MedianBlur)
- [MotionBlur](https://explore.albumentations.ai/transform/MotionBlur)
- [MultiplicativeNoise](https://explore.albumentations.ai/transform/MultiplicativeNoise)
- [Normalize](https://explore.albumentations.ai/transform/Normalize)
- [PixelDistributionAdaptation](https://explore.albumentations.ai/transform/PixelDistributionAdaptation)
- [PlanckianJitter](https://explore.albumentations.ai/transform/PlanckianJitter)
- [PlasmaBrightnessContrast](https://explore.albumentations.ai/transform/PlasmaBrightnessContrast)
- [PlasmaShadow](https://explore.albumentations.ai/transform/PlasmaShadow)
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
- [SaltAndPepper](https://explore.albumentations.ai/transform/SaltAndPepper)
- [Sharpen](https://explore.albumentations.ai/transform/Sharpen)
- [ShotNoise](https://explore.albumentations.ai/transform/ShotNoise)
- [Solarize](https://explore.albumentations.ai/transform/Solarize)
- [Spatter](https://explore.albumentations.ai/transform/Spatter)
- [Superpixels](https://explore.albumentations.ai/transform/Superpixels)
- [TextImage](https://explore.albumentations.ai/transform/TextImage)
- [ToFloat](https://explore.albumentations.ai/transform/ToFloat)
- [ToGray](https://explore.albumentations.ai/transform/ToGray)
- [ToRGB](https://explore.albumentations.ai/transform/ToRGB)
- [ToSepia](https://explore.albumentations.ai/transform/ToSepia)
- [UnsharpMask](https://explore.albumentations.ai/transform/UnsharpMask)
- [ZoomBlur](https://explore.albumentations.ai/transform/ZoomBlur)

### Spatial-level transforms

Spatial-level transforms will simultaneously change both an input image as well as additional targets such as masks, bounding boxes, and keypoints. For volumetric data (volumes and 3D masks), these transforms are applied independently to each slice along the Z-axis (depth dimension), maintaining consistency across the volume. The following table shows which additional targets are supported by each transform:

- Volume: 3D array of shape (D, H, W) or (D, H, W, C) where D is depth, H is height, W is width, and C is number of channels (optional)
- Mask3D: Binary or multi-class 3D mask of shape (D, H, W) where each slice represents segmentation for the corresponding volume slice

| Transform                                                                                        | Image | Mask | BBoxes | Keypoints | Volume | Mask3D |
| ------------------------------------------------------------------------------------------------ | :---: | :--: | :----: | :-------: | :----: | :----: |
| [Affine](https://explore.albumentations.ai/transform/Affine)                                     | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [AtLeastOneBBoxRandomCrop](https://explore.albumentations.ai/transform/AtLeastOneBBoxRandomCrop) | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [BBoxSafeRandomCrop](https://explore.albumentations.ai/transform/BBoxSafeRandomCrop)             | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [CenterCrop](https://explore.albumentations.ai/transform/CenterCrop)                             | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [CoarseDropout](https://explore.albumentations.ai/transform/CoarseDropout)                       | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [ConstrainedCoarseDropout](https://explore.albumentations.ai/transform/ConstrainedCoarseDropout) | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [Crop](https://explore.albumentations.ai/transform/Crop)                                         | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [CropAndPad](https://explore.albumentations.ai/transform/CropAndPad)                             | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [CropNonEmptyMaskIfExists](https://explore.albumentations.ai/transform/CropNonEmptyMaskIfExists) | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [D4](https://explore.albumentations.ai/transform/D4)                                             | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [ElasticTransform](https://explore.albumentations.ai/transform/ElasticTransform)                 | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [Erasing](https://explore.albumentations.ai/transform/Erasing)                                   | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [FrequencyMasking](https://explore.albumentations.ai/transform/FrequencyMasking)                 | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [GridDistortion](https://explore.albumentations.ai/transform/GridDistortion)                     | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [GridDropout](https://explore.albumentations.ai/transform/GridDropout)                           | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [GridElasticDeform](https://explore.albumentations.ai/transform/GridElasticDeform)               | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [HorizontalFlip](https://explore.albumentations.ai/transform/HorizontalFlip)                     | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [Lambda](https://explore.albumentations.ai/transform/Lambda)                                     | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [LongestMaxSize](https://explore.albumentations.ai/transform/LongestMaxSize)                     | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [MaskDropout](https://explore.albumentations.ai/transform/MaskDropout)                           | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [Morphological](https://explore.albumentations.ai/transform/Morphological)                       | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [Mosaic](https://explore.albumentations.ai/transform/Mosaic)                                     | ✓     | ✓    | ✓      | ✓         |        |        |
| [NoOp](https://explore.albumentations.ai/transform/NoOp)                                         | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [OpticalDistortion](https://explore.albumentations.ai/transform/OpticalDistortion)               | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [OverlayElements](https://explore.albumentations.ai/transform/OverlayElements)                   | ✓     | ✓    |        |           |        |        |
| [Pad](https://explore.albumentations.ai/transform/Pad)                                           | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [PadIfNeeded](https://explore.albumentations.ai/transform/PadIfNeeded)                           | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [Perspective](https://explore.albumentations.ai/transform/Perspective)                           | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [PiecewiseAffine](https://explore.albumentations.ai/transform/PiecewiseAffine)                   | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [PixelDropout](https://explore.albumentations.ai/transform/PixelDropout)                         | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [RandomCrop](https://explore.albumentations.ai/transform/RandomCrop)                             | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [RandomCropFromBorders](https://explore.albumentations.ai/transform/RandomCropFromBorders)       | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [RandomCropNearBBox](https://explore.albumentations.ai/transform/RandomCropNearBBox)             | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [RandomGridShuffle](https://explore.albumentations.ai/transform/RandomGridShuffle)               | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [RandomResizedCrop](https://explore.albumentations.ai/transform/RandomResizedCrop)               | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [RandomRotate90](https://explore.albumentations.ai/transform/RandomRotate90)                     | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [RandomScale](https://explore.albumentations.ai/transform/RandomScale)                           | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [RandomSizedBBoxSafeCrop](https://explore.albumentations.ai/transform/RandomSizedBBoxSafeCrop)   | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [RandomSizedCrop](https://explore.albumentations.ai/transform/RandomSizedCrop)                   | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [Resize](https://explore.albumentations.ai/transform/Resize)                                     | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [Rotate](https://explore.albumentations.ai/transform/Rotate)                                     | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [SafeRotate](https://explore.albumentations.ai/transform/SafeRotate)                             | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [ShiftScaleRotate](https://explore.albumentations.ai/transform/ShiftScaleRotate)                 | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [SmallestMaxSize](https://explore.albumentations.ai/transform/SmallestMaxSize)                   | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [SquareSymmetry](https://explore.albumentations.ai/transform/SquareSymmetry)                     | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [ThinPlateSpline](https://explore.albumentations.ai/transform/ThinPlateSpline)                   | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [TimeMasking](https://explore.albumentations.ai/transform/TimeMasking)                           | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [TimeReverse](https://explore.albumentations.ai/transform/TimeReverse)                           | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [Transpose](https://explore.albumentations.ai/transform/Transpose)                               | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [VerticalFlip](https://explore.albumentations.ai/transform/VerticalFlip)                         | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |
| [XYMasking](https://explore.albumentations.ai/transform/XYMasking)                               | ✓     | ✓    | ✓      | ✓         | ✓      | ✓      |

### 3D transforms

3D transforms operate on volumetric data and can modify both the input volume and associated 3D mask.

Where:

- Volume: 3D array of shape (D, H, W) or (D, H, W, C) where D is depth, H is height, W is width, and C is number of channels (optional)
- Mask3D: Binary or multi-class 3D mask of shape (D, H, W) where each slice represents segmentation for the corresponding volume slice

| Transform                                                                      | Volume | Mask3D | Keypoints |
| ------------------------------------------------------------------------------ | :----: | :----: | :-------: |
| [CenterCrop3D](https://explore.albumentations.ai/transform/CenterCrop3D)       | ✓      | ✓      | ✓         |
| [CoarseDropout3D](https://explore.albumentations.ai/transform/CoarseDropout3D) | ✓      | ✓      | ✓         |
| [CubicSymmetry](https://explore.albumentations.ai/transform/CubicSymmetry)     | ✓      | ✓      | ✓         |
| [Pad3D](https://explore.albumentations.ai/transform/Pad3D)                     | ✓      | ✓      | ✓         |
| [PadIfNeeded3D](https://explore.albumentations.ai/transform/PadIfNeeded3D)     | ✓      | ✓      | ✓         |
| [RandomCrop3D](https://explore.albumentations.ai/transform/RandomCrop3D)       | ✓      | ✓      | ✓         |

## A few more examples of **augmentations**

### Semantic segmentation on the Inria dataset

![inria](https://habrastorage.org/webt/su/wa/np/suwanpeo6ww7wpwtobtrzd_cg20.jpeg)

### Medical imaging

![medical](https://habrastorage.org/webt/1i/fi/wz/1ifiwzy0lxetc4nwjvss-71nkw0.jpeg)

### Object detection and semantic segmentation on the Mapillary Vistas dataset

![vistas](https://habrastorage.org/webt/rz/-h/3j/rz-h3jalbxic8o_fhucxysts4tc.jpeg)

### Keypoints augmentation

<img src="https://habrastorage.org/webt/e-/6k/z-/e-6kz-fugp2heak3jzns3bc-r8o.jpeg" width=100%>

## Benchmark Results

### Image Benchmark Results

### System Information

- Platform: macOS-15.1-arm64-arm-64bit
- Processor: arm
- CPU Count: 16
- Python Version: 3.12.8

### Benchmark Parameters

- Number of images: 2000
- Runs per transform: 5
- Max warmup iterations: 1000

### Library Versions

- albumentations: 2.0.4
- augly: 1.0.0
- imgaug: 0.4.0
- kornia: 0.8.0
- torchvision: 0.20.1

## Performance Comparison

Number shows how many uint8 images per second can be processed on one CPU thread. Larger is better.
The Speedup column shows how many times faster Albumentations is compared to the fastest other
library for each transform.

| Transform            | albumentations<br>2.0.4   | augly<br>1.0.0   | imgaug<br>0.4.0   | kornia<br>0.8.0   | torchvision<br>0.20.1   | Speedup<br>(Alb/fastest other)   |
|:---------------------|:--------------------------|:-----------------|:------------------|:------------------|:------------------------|:---------------------------------|
| Affine               | **1445 ± 9**              | -                | 1328 ± 16         | 248 ± 6           | 188 ± 2                 | 1.09x                            |
| AutoContrast         | **1657 ± 13**             | -                | -                 | 541 ± 8           | 344 ± 1                 | 3.06x                            |
| Blur                 | **7657 ± 114**            | 386 ± 4          | 5381 ± 125        | 265 ± 11          | -                       | 1.42x                            |
| Brightness           | **11985 ± 455**           | 2108 ± 32        | 1076 ± 32         | 1127 ± 27         | 854 ± 13                | 5.68x                            |
| CLAHE                | **647 ± 4**               | -                | 555 ± 14          | 165 ± 3           | -                       | 1.17x                            |
| CenterCrop128        | **119293 ± 2164**         | -                | -                 | -                 | -                       | N/A                              |
| ChannelDropout       | **11534 ± 306**           | -                | -                 | 2283 ± 24         | -                       | 5.05x                            |
| ChannelShuffle       | **6772 ± 109**            | -                | 1252 ± 26         | 1328 ± 44         | 4417 ± 234              | 1.53x                            |
| CoarseDropout        | **18962 ± 1346**          | -                | 1190 ± 22         | -                 | -                       | 15.93x                           |
| ColorJitter          | **1020 ± 91**             | 418 ± 5          | -                 | 104 ± 4           | 87 ± 1                  | 2.44x                            |
| Contrast             | **12394 ± 363**           | 1379 ± 25        | 717 ± 5           | 1109 ± 41         | 602 ± 13                | 8.99x                            |
| CornerIllumination   | **484 ± 7**               | -                | -                 | 452 ± 3           | -                       | 1.07x                            |
| Elastic              | 374 ± 2                   | -                | **395 ± 14**      | 1 ± 0             | 3 ± 0                   | 0.95x                            |
| Equalize             | **1236 ± 21**             | -                | 814 ± 11          | 306 ± 1           | 795 ± 3                 | 1.52x                            |
| Erasing              | **27451 ± 2794**          | -                | -                 | 1210 ± 27         | 3577 ± 49               | 7.67x                            |
| GaussianBlur         | **2350 ± 118**            | 387 ± 4          | 1460 ± 23         | 254 ± 5           | 127 ± 4                 | 1.61x                            |
| GaussianIllumination | **720 ± 7**               | -                | -                 | 436 ± 13          | -                       | 1.65x                            |
| GaussianNoise        | **315 ± 4**               | -                | 263 ± 9           | 125 ± 1           | -                       | 1.20x                            |
| Grayscale            | **32284 ± 1130**          | 6088 ± 107       | 3100 ± 24         | 1201 ± 52         | 2600 ± 23               | 5.30x                            |
| HSV                  | **1197 ± 23**             | -                | -                 | -                 | -                       | N/A                              |
| HorizontalFlip       | **14460 ± 368**           | 8808 ± 1012      | 9599 ± 495        | 1297 ± 13         | 2486 ± 107              | 1.51x                            |
| Hue                  | **1944 ± 64**             | -                | -                 | 150 ± 1           | -                       | 12.98x                           |
| Invert               | **27665 ± 3803**          | -                | 3682 ± 79         | 2881 ± 43         | 4244 ± 30               | 6.52x                            |
| JpegCompression      | **1321 ± 33**             | 1202 ± 19        | 687 ± 26          | 120 ± 1           | 889 ± 7                 | 1.10x                            |
| LinearIllumination   | 479 ± 5                   | -                | -                 | **708 ± 6**       | -                       | 0.68x                            |
| MedianBlur           | **1229 ± 9**              | -                | 1152 ± 14         | 6 ± 0             | -                       | 1.07x                            |
| MotionBlur           | **3521 ± 25**             | -                | 928 ± 37          | 159 ± 1           | -                       | 3.79x                            |
| Normalize            | **1819 ± 49**             | -                | -                 | 1251 ± 14         | 1018 ± 7                | 1.45x                            |
| OpticalDistortion    | **661 ± 7**               | -                | -                 | 174 ± 0           | -                       | 3.80x                            |
| Pad                  | **48589 ± 2059**          | -                | -                 | -                 | 4889 ± 183              | 9.94x                            |
| Perspective          | **1206 ± 3**              | -                | 908 ± 8           | 154 ± 3           | 147 ± 5                 | 1.33x                            |
| PlankianJitter       | **3221 ± 63**             | -                | -                 | 2150 ± 52         | -                       | 1.50x                            |
| PlasmaBrightness     | **168 ± 2**               | -                | -                 | 85 ± 1            | -                       | 1.98x                            |
| PlasmaContrast       | **145 ± 3**               | -                | -                 | 84 ± 0            | -                       | 1.71x                            |
| PlasmaShadow         | 183 ± 5                   | -                | -                 | **216 ± 5**       | -                       | 0.85x                            |
| Posterize            | **12979 ± 1121**          | -                | 3111 ± 95         | 836 ± 30          | 4247 ± 26               | 3.06x                            |
| RGBShift             | **3391 ± 104**            | -                | -                 | 896 ± 9           | -                       | 3.79x                            |
| Rain                 | **2043 ± 115**            | -                | -                 | 1493 ± 9          | -                       | 1.37x                            |
| RandomCrop128        | **111859 ± 1374**         | 45395 ± 934      | 21408 ± 622       | 2946 ± 42         | 31450 ± 249             | 2.46x                            |
| RandomGamma          | **12444 ± 753**           | -                | 3504 ± 72         | 230 ± 3           | -                       | 3.55x                            |
| RandomResizedCrop    | **4347 ± 37**             | -                | -                 | 661 ± 16          | 837 ± 37                | 5.19x                            |
| Resize               | **3532 ± 67**             | 1083 ± 21        | 2995 ± 70         | 645 ± 13          | 260 ± 9                 | 1.18x                            |
| Rotate               | **2912 ± 68**             | 1739 ± 105       | 2574 ± 10         | 256 ± 2           | 258 ± 4                 | 1.13x                            |
| SaltAndPepper        | **629 ± 6**               | -                | -                 | 480 ± 12          | -                       | 1.31x                            |
| Saturation           | **1596 ± 24**             | -                | 495 ± 3           | 155 ± 2           | -                       | 3.22x                            |
| Sharpen              | **2346 ± 10**             | -                | 1101 ± 30         | 201 ± 2           | 220 ± 3                 | 2.13x                            |
| Shear                | **1299 ± 11**             | -                | 1244 ± 14         | 261 ± 1           | -                       | 1.04x                            |
| Snow                 | **611 ± 9**               | -                | -                 | 143 ± 1           | -                       | 4.28x                            |
| Solarize             | **11756 ± 481**           | -                | 3843 ± 80         | 263 ± 6           | 1032 ± 14               | 3.06x                            |
| ThinPlateSpline      | **82 ± 1**                | -                | -                 | 58 ± 0            | -                       | 1.41x                            |
| VerticalFlip         | **32386 ± 936**           | 16830 ± 1653     | 19935 ± 1708      | 2872 ± 37         | 4696 ± 161              | 1.62x                            |

## Contributing

To create a pull request to the repository, follow the documentation at [CONTRIBUTING.md](CONTRIBUTING.md)

![https://github.com/albuemntations-team/albumentation/graphs/contributors](https://contrib.rocks/image?repo=albumentations-team/albumentations)

## Community

- [LinkedIn](https://www.linkedin.com/company/albumentations/)
- [Twitter](https://twitter.com/albumentations)
- [Discord](https://discord.gg/AKPrrDYNAt)

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
