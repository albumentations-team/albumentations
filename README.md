# Albumentations
[![Build Status](https://travis-ci.org/albu/albumentations.svg?branch=master)](https://travis-ci.org/albu/albumentations)
[![Documentation Status](https://readthedocs.org/projects/albumentations/badge/?version=latest)](https://albumentations.readthedocs.io/en/latest/?badge=latest)

* Great fast augmentations based on highly-optimized OpenCV library
* Super simple yet powerful interface for different tasks like (segmentation, detection, etc.)
* Easy to customize
* Easy to add other frameworks


## How to use
**Classification** - [`example.ipynb`](notebooks/example.ipynb)

**Object detection** - [`example_bboxes.ipynb`](notebooks/example_bboxes.ipynb)

**Non-8-bit images** - [`example_16_bit_tiff.ipynb`](notebooks/example_16_bit_tiff.ipynb)

**Image segmentation** is also supported out of the box, notebook coming soon.

**Custom tasks** such as autoencoders, more then three channel images - [`custom_targets`](https://gist.github.com/poxyu/799b359d7f87205b7c4288691ab019dc)

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


## Migrating from torchvision to albumentations

Migrating from torchvision to albumentations is simple - you just need to change a few lines of code. 
Albumentations has equivalents for common torchvision transforms as well as plenty of transforms that are not presented in torchvision.
[`migrating_from_torchvision_to_albumentations.ipynb`](notebooks/migrating_from_torchvision_to_albumentations.ipynb) shows how one can migrate code from torchvision to albumentations.


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


# Citing

If you find this library useful for your research, please consider citing:

```
@article{2018arXiv180906839B,
    author = {A. Buslaev, A. Parinov, E. Khvedchenya, V.~I. Iglovikov and 
	A.~A. Kalinin},
     title = "{Albumentations: fast and flexible image augmentations}",
   journal = {ArXiv e-prints},
    eprint = {1809.06839}, 
      year = 2018      
}
```



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
