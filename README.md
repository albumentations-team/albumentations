# Albumentations
[![PyPI version](https://badge.fury.io/py/albumentations.svg)](https://badge.fury.io/py/albumentations)
![CI](https://github.com/albumentations-team/albumentations/workflows/CI/badge.svg)

Albumentations is a Python library for image augmentation. Image augmentation is used in deep learning and computer vision tasks to increase the quality of trained models. The purpose of image augmentation is to create new training samples from the existing data.

Here is an example of how you can apply some augmentations from Albumentations to create new images from the original one:
![parrot](https://habrastorage.org/webt/bd/ne/rv/bdnerv5ctkudmsaznhw4crsdfiw.jpeg)

## Why Albumentations
- Albumentations **supports all common computer vision tasks** such as classification, semantic segmentation, instance segmentation, object detection, and pose estimation.
- The library provides **a simple unified API** to work with all data types: images (RBG-images, grayscale images, multispectral images), segmentation masks, bounding boxes, and keypoints.
- The library contains **more than 70 different augmentations** to generate new training samples from the existing data.
- Albumentations is **fast**. We benchmark each new release to ensure that augmentations provide maximum speed.
- It **works with popular deep learning frameworks** such as PyTorch and TensorFlow 2. By the way, Albumentations is a part of the PyTorch ecosystem.
- **Written by experts**. The authors have experience both working on production computer vision systems and participating in competitive machine learning. Many core team members are Kaggle Masters and Grandmasters.
- The library is **widely used** in industry, deep learning research, machine learning competitions, and open source projects.

## Table of contents
- [Authors](#authors)
- [Installation](#installation)
- [Documentation](#documentation)
- [Getting started](#getting-started)
  - [I am new to image augmentation](#i-am-new-to-image-augmentation)
  - [I want to use Albumentations for the specific task such as classification or segmentation](#i-want-to-use-albumentations-for-the-specific-task-such-as-classification-or-segmentation)
  - [I want to explore augmentations and see Albumentations in action](#i-want-to-explore-augmentations-and-see-albumentations-in-action)
- [Who is using Albumentations](#who-is-using-albumentations)
- [A few more examples of augmentations](#a-few-more-examples-of-augmentations)
- [Benchmarking results](#benchmarking-results)
- [Contributing](#contributing)
- [Comments](#comments)
- [Citing](#citing)

## Authors
[Alexander Buslaev](https://www.linkedin.com/in/al-buslaev/)

[Alex Parinov](https://www.linkedin.com/in/alex-parinov/)

[Vladimir I. Iglovikov](https://www.linkedin.com/in/iglovikov/)

[Evegene Khvedchenya](https://www.linkedin.com/in/cvtalks/)

[Mikhail Druzhinin](https://www.linkedin.com/in/mikhail-druzhinin-548229100/)


## Installation
Albumentations requires Python 3.6 or higher. To install the latest version from PyPI:

```
pip install -U albumentations
```

Other installation options are described in the [documentation](https://albumentations.ai/docs/getting_started/installation/)


## Documentation
The full documentation is available at **[https://albumentations.ai/docs/](https://albumentations.ai/docs/)**.

## Getting started

### I am new to image augmentation
Please start with the [introduction articles](https://albumentations.ai/docs/#introduction-to-image-augmentation) about why image augmentation is important and why it helps to build better models.

### I want to use Albumentations for the specific task such as classification or segmentation
If you want to use Albumentations for a specific task such as classification, segmentation, or object detection, refer to the [set of articles](https://albumentations.ai/docs/#getting-started-with-albumentations) that has an in-depth description of this task. We also have a [list of examples](https://albumentations.ai/docs/examples/) of how to use Albumentations for different use cases and integrate it with deep learning frameworks such as PyTorch and TensorFlow 2.

### I want to explore augmentations and see Albumentations in action
Check the [online demo of the library](https://albumentations-demo.herokuapp.com/). With it, you can apply augmentations to different images and see the result. Also, we have a [special page](https://albumentations.ai/docs/getting_started/transforms_and_targets/) that lists all available augmentations and their targets.


## Who is using Albumentations
<a href="https://www.lyft.com/" target="_blank"><img src="https://habrastorage.org/webt/ce/bs/sa/cebssajf_5asst5yshmyykqjhcg.png" width="100"/></a>
<a href="https://www.x5.ru/en" target="_blank"><img src="https://habrastorage.org/webt/9y/dv/f1/9ydvf1fbxotkl6nyhydrn9v8cqw.png" width="100"/></a>
<a href="https://imedhub.org/" target="_blank"><img src="https://habrastorage.org/webt/eq/8x/m-/eq8xm-fjfx_uqkka4_ekxsdwtiq.png" width="100"/></a>
<a href="https://recursionpharma.com" target="_blank"><img src="https://pbs.twimg.com/profile_images/925897897165639683/jI8YvBfC_400x400.jpg" width="100"/></a>
<a href="https://www.everypixel.com/" target="_blank"><img src="https://www.everypixel.com/i/logo_sq.png" width="100"/></a>
<a href="https://neuromation.io/" target="_blank"><img src="https://habrastorage.org/webt/yd/_4/xa/yd_4xauvggn1tuz5xgrtkif6lya.png" width="100"/></a>

#### See also:
- [A list of papers that cite Albumentations]().
- [A list of teams that were using Albumentations and took high places in machine learning competitions]().
- [Open source projects that use Albumentations]().

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
The table shows how many images per second can be processed on a single core, higher is better.

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
