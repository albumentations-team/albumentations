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

- **Complete Computer Vision Support**: Works with [all major CV tasks](#i-want-to-use-albumentations-for-the-specific-task-such-as-classification-or-segmentation) including classification, segmentation (semantic & instance), object detection, and pose estimation.
- **Simple, Unified API**: [One consistent interface](#a-simple-example) for all data types - RGB/grayscale/multispectral images, masks, bounding boxes, and keypoints.
- **Rich Augmentation Library**: [70+ high-quality augmentations](https://albumentations.ai/docs/api_reference/transforms/) to enhance your training data.
- **Fast**: Consistently benchmarked as the [fastest augmentation library](https://albumentations.ai/docs/benchmarks/), with optimizations for production use.
- **Deep Learning Integration**: Works with [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/), and other frameworks. Part of the [PyTorch ecosystem](https://pytorch.org/ecosystem/).
- **Created by Experts**: Built by [developers with deep experience in computer vision and machine learning competitions](https://albumentations.ai/docs/#authors).
- **Widely Used**: Applied in [research, competitions, and production systems](https://albumentations.ai/whos_using) across many organizations.

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
  - [Benchmarking results](#benchmarking-results)
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

Please start with the [introduction articles](https://albumentations.ai/docs/#introduction-to-image-augmentation) about why image augmentation is important and how it helps to build better models.

### I want to use Albumentations for the specific task such as classification or segmentation

If you want to use Albumentations for a specific task such as classification, segmentation, or object detection, refer to the [set of articles](https://albumentations.ai/docs/#getting-started-with-albumentations) that has an in-depth description of this task. We also have a [list of examples](https://albumentations.ai/docs/examples/) on applying Albumentations for different use cases.

### I want to know how to use Albumentations with deep learning frameworks

We have [examples of using Albumentations](https://albumentations.ai/docs/#examples-of-how-to-use-albumentations-with-different-deep-learning-frameworks) along with PyTorch and TensorFlow.

### I want to explore augmentations and see Albumentations in action

Check the [online demo of the library](https://albumentations-demo.herokuapp.com/). With it, you can apply augmentations to different images and see the result. Also, we have a [list of all available augmentations and their targets](#list-of-augmentations).

## Who is using Albumentations

<a href="https://www.apple.com/" target="_blank"><img src="https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/website/public/assets/industry/apple.jpeg" width="100"/></a>
<a href="https://research.google/" target="_blank"><img src="https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/website/public/assets/industry/google.png" width="100"/></a>
<a href="https://opensource.fb.com/" target="_blank"><img src="https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/website/public/assets/industry/meta_research.png" width="100"/></a>
<a href="https: //www.nvidia.com/en-us/research/" target="_blank"><img src="https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/website/public/assets/industry/nvidia_research.jpeg" width="100"/></a>
<a href="https://www.amazon.science/" target="_blank"><img src="https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/website/public/assets/industry/amazon_science.png" width="100"/></a>
<a href="https://opensource.microsoft.com/" target="_blank"><img src="https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/website/public/assets/industry/microsoft.png" width="100"/></a>
<a href="https://engineering.salesforce.com/open-source/" target="_blank"><img src="https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/website/public/assets/industry/salesforce_open_source.png" width="100"/></a>
<a href="https://stability.ai/" target="_blank"><img src="https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/website/public/assets/industry/stability.png" width="100"/></a>
<a href="https://www.ibm.com/opensource/" target="_blank"><img src="https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/website/public/assets/industry/ibm.jpeg" width="100"/></a>
<a href="https://huggingface.co/" target="_blank"><img src="https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/website/public/assets/industry/hugging_face.png" width="100"/></a>
<a href="https://www.sony.com/en/" target="_blank"><img src="https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/website/public/assets/industry/sony.png" width="100"/></a>
<a href="https://opensource.alibaba.com/" target="_blank"><img src="https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/website/public/assets/industry/alibaba.png" width="100"/></a>
<a href="https://opensource.tencent.com/" target="_blank"><img src="https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/website/public/assets/industry/tencent.png" width="100"/></a>
<a href="https://h2o.ai/" target="_blank"><img src="https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/website/public/assets/industry/h2o_ai.png" width="100"/></a>

### See also

- [A list of papers that cite Albumentations](https://scholar.google.com/citations?view_op=view_citation&citation_for_view=vkjh9X0AAAAJ:r0BpntZqJG4C).
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
- [RandomGrayscale](https://explore.albumentations.ai/transform/RandomGrayscale)
- [RandomJPEG](https://explore.albumentations.ai/transform/RandomJPEG)
- [RandomRain](https://explore.albumentations.ai/transform/RandomRain)
- [RandomShadow](https://explore.albumentations.ai/transform/RandomShadow)
- [RandomSnow](https://explore.albumentations.ai/transform/RandomSnow)
- [RandomSunFlare](https://explore.albumentations.ai/transform/RandomSunFlare)
- [RandomToneCurve](https://explore.albumentations.ai/transform/RandomToneCurve)
- [RingingOvershoot](https://explore.albumentations.ai/transform/RingingOvershoot)
- [Sharpen](https://explore.albumentations.ai/transform/Sharpen)
- [ShotNoise](https://explore.albumentations.ai/transform/ShotNoise)
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
| [Erasing](https://explore.albumentations.ai/transform/Erasing)                                   | ✓     | ✓    | ✓      | ✓         |
| [FrequencyMasking](https://explore.albumentations.ai/transform/FrequencyMasking)                 | ✓     | ✓    | ✓      | ✓         |
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
| [Pad](https://explore.albumentations.ai/transform/Pad)                                           | ✓     | ✓    | ✓      | ✓         |
| [PadIfNeeded](https://explore.albumentations.ai/transform/PadIfNeeded)                           | ✓     | ✓    | ✓      | ✓         |
| [Perspective](https://explore.albumentations.ai/transform/Perspective)                           | ✓     | ✓    | ✓      | ✓         |
| [PiecewiseAffine](https://explore.albumentations.ai/transform/PiecewiseAffine)                   | ✓     | ✓    | ✓      | ✓         |
| [PixelDropout](https://explore.albumentations.ai/transform/PixelDropout)                         | ✓     | ✓    | ✓      | ✓         |
| [RandomAffine](https://explore.albumentations.ai/transform/RandomAffine)                         | ✓     | ✓    | ✓      | ✓         |
| [RandomCrop](https://explore.albumentations.ai/transform/RandomCrop)                             | ✓     | ✓    | ✓      | ✓         |
| [RandomCropFromBorders](https://explore.albumentations.ai/transform/RandomCropFromBorders)       | ✓     | ✓    | ✓      | ✓         |
| [RandomErasing](https://explore.albumentations.ai/transform/RandomErasing)                       | ✓     | ✓    | ✓      | ✓         |
| [RandomGridShuffle](https://explore.albumentations.ai/transform/RandomGridShuffle)               | ✓     | ✓    | ✓      | ✓         |
| [RandomHorizontalFlip](https://explore.albumentations.ai/transform/RandomHorizontalFlip)         | ✓     | ✓    | ✓      | ✓         |
| [RandomPerspective](https://explore.albumentations.ai/transform/RandomPerspective)               | ✓     | ✓    | ✓      | ✓         |
| [RandomResizedCrop](https://explore.albumentations.ai/transform/RandomResizedCrop)               | ✓     | ✓    | ✓      | ✓         |
| [RandomRotate90](https://explore.albumentations.ai/transform/RandomRotate90)                     | ✓     | ✓    | ✓      | ✓         |
| [RandomRotation](https://explore.albumentations.ai/transform/RandomRotation)                     | ✓     | ✓    | ✓      | ✓         |
| [RandomScale](https://explore.albumentations.ai/transform/RandomScale)                           | ✓     | ✓    | ✓      | ✓         |
| [RandomSizedBBoxSafeCrop](https://explore.albumentations.ai/transform/RandomSizedBBoxSafeCrop)   | ✓     | ✓    | ✓      | ✓         |
| [RandomSizedCrop](https://explore.albumentations.ai/transform/RandomSizedCrop)                   | ✓     | ✓    | ✓      | ✓         |
| [RandomVerticalFlip](https://explore.albumentations.ai/transform/RandomVerticalFlip)             | ✓     | ✓    | ✓      | ✓         |
| [Resize](https://explore.albumentations.ai/transform/Resize)                                     | ✓     | ✓    | ✓      | ✓         |
| [Rotate](https://explore.albumentations.ai/transform/Rotate)                                     | ✓     | ✓    | ✓      | ✓         |
| [SafeRotate](https://explore.albumentations.ai/transform/SafeRotate)                             | ✓     | ✓    | ✓      | ✓         |
| [ShiftScaleRotate](https://explore.albumentations.ai/transform/ShiftScaleRotate)                 | ✓     | ✓    | ✓      | ✓         |
| [SmallestMaxSize](https://explore.albumentations.ai/transform/SmallestMaxSize)                   | ✓     | ✓    | ✓      | ✓         |
| [TimeMasking](https://explore.albumentations.ai/transform/TimeMasking)                           | ✓     | ✓    | ✓      | ✓         |
| [TimeReverse](https://explore.albumentations.ai/transform/TimeReverse)                           | ✓     | ✓    | ✓      | ✓         |
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

### System Information

- Platform: macOS-15.0.1-arm64-arm-64bit
- Processor: arm
- CPU Count: 10
- Python Version: 3.12.7

### Benchmark Parameters

- Number of images: 1000
- Runs per transform: 10
- Max warmup iterations: 1000

### Library Versions

- albumentations: 1.4.20
- augly: 1.0.0
- imgaug: 0.4.0
- kornia: 0.7.3
- torchvision: 0.20.0

## Performance Comparison

Number - is the number of uint8 RGB images processed per second on a single CPU core. Higher is better.

| Transform         | albumentations<br>1.4.20   | augly<br>1.0.0   | imgaug<br>0.4.0   | kornia<br>0.7.3   | torchvision<br>0.20.0   |
|:------------------|:---------------------------|:-----------------|:------------------|:------------------|:------------------------|
| HorizontalFlip    | **8618 ± 1233**            | 4807 ± 818       | 6042 ± 788        | 390 ± 106         | 914 ± 67                |
| VerticalFlip      | **22847 ± 2031**           | 9153 ± 1291      | 10931 ± 1844      | 1212 ± 402        | 3198 ± 200              |
| Rotate            | **1146 ± 79**              | 1119 ± 41        | 1136 ± 218        | 143 ± 11          | 181 ± 11                |
| Affine            | 682 ± 192                  | -                | **774 ± 97**      | 147 ± 9           | 130 ± 12                |
| Equalize          | **892 ± 61**               | -                | 581 ± 54          | 152 ± 19          | 479 ± 12                |
| RandomCrop80      | **47341 ± 20523**          | 25272 ± 1822     | 11503 ± 441       | 1510 ± 230        | 32109 ± 1241            |
| ShiftRGB          | **2349 ± 76**              | -                | 1582 ± 65         | -                 | -                       |
| Resize            | **2316 ± 166**             | 611 ± 78         | 1806 ± 63         | 232 ± 24          | 195 ± 4                 |
| RandomGamma       | **8675 ± 274**             | -                | 2318 ± 269        | 108 ± 13          | -                       |
| Grayscale         | **3056 ± 47**              | 2720 ± 932       | 1681 ± 156        | 289 ± 75          | 1838 ± 130              |
| RandomPerspective | 412 ± 38                   | -                | **554 ± 22**      | 86 ± 11           | 96 ± 5                  |
| GaussianBlur      | **1728 ± 89**              | 242 ± 4          | 1090 ± 65         | 176 ± 18          | 79 ± 3                  |
| MedianBlur        | **868 ± 60**               | -                | 813 ± 30          | 5 ± 0             | -                       |
| MotionBlur        | **4047 ± 67**              | -                | 612 ± 18          | 73 ± 2            | -                       |
| Posterize         | **9094 ± 301**             | -                | 2097 ± 68         | 430 ± 49          | 3196 ± 185              |
| JpegCompression   | **918 ± 23**               | 778 ± 5          | 459 ± 35          | 71 ± 3            | 625 ± 17                |
| GaussianNoise     | 166 ± 12                   | 67 ± 2           | **206 ± 11**      | 75 ± 1            | -                       |
| Elastic           | 201 ± 5                    | -                | **235 ± 20**      | 1 ± 0             | 2 ± 0                   |
| Clahe             | **454 ± 22**               | -                | 335 ± 43          | 94 ± 9            | -                       |
| CoarseDropout     | **13368 ± 744**            | -                | 671 ± 38          | 536 ± 87          | -                       |
| Blur              | **5267 ± 543**             | 246 ± 3          | 3807 ± 325        | -                 | -                       |
| ColorJitter       | **628 ± 55**               | 255 ± 13         | -                 | 55 ± 18           | 46 ± 2                  |
| Brightness        | **8956 ± 300**             | 1163 ± 86        | -                 | 472 ± 101         | 429 ± 20                |
| Contrast          | **8879 ± 1426**            | 736 ± 79         | -                 | 425 ± 52          | 335 ± 35                |
| RandomResizedCrop | **2828 ± 186**             | -                | -                 | 287 ± 58          | 511 ± 10                |
| Normalize         | **1196 ± 56**              | -                | -                 | 626 ± 40          | 519 ± 12                |
| PlankianJitter    | **2204 ± 385**             | -                | -                 | 813 ± 211         | -                       |

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
