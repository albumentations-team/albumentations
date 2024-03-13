import argparse
import os
import random
import sys
from collections import defaultdict
from contextlib import suppress
from pathlib import Path
from timeit import Timer
from typing import Dict, List, Union

import augly.image as imaugs
import cv2
import kornia as K
import kornia.augmentation as Kaug
import numpy as np
import pandas as pd
import pkg_resources
import tensorflow as tf
import torch
import torchvision.transforms.functional as torchvision
from Augmentor import Operations, Pipeline
from imgaug import augmenters as iaa
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as T
from tqdm import tqdm

import albumentations as A
from benchmark.utils import (
    MarkdownGenerator,
    format_results,
    read_img_cv2,
    read_img_kornia,
    read_img_pillow,
    read_img_tensorflow,
    read_img_torch,
)

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


DEFAULT_BENCHMARKING_LIBRARIES = [
    "albumentations",
    "kornia",
    "torchvision",
    "tensorflow",
    "imgaug",
    "augly",
    "augmentor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augmentation libraries performance benchmark")
    parser.add_argument(
        "-d", "--data-dir", metavar="DIR", default=os.environ.get("DATA_DIR"), help="path to a directory with images"
    )
    parser.add_argument(
        "-i", "--images", default=2000, type=int, metavar="N", help="number of images for benchmarking (default: 2000)"
    )
    parser.add_argument(
        "-l", "--libraries", default=DEFAULT_BENCHMARKING_LIBRARIES, nargs="+", help="list of libraries to benchmark"
    )
    parser.add_argument(
        "-r", "--runs", default=5, type=int, metavar="N", help="number of runs for each benchmark (default: 5)"
    )
    parser.add_argument(
        "--show-std", dest="show_std", action="store_true", help="show standard deviation for benchmark runs"
    )
    parser.add_argument("-p", "--print-package-versions", action="store_true", help="print versions of packages")
    parser.add_argument("-m", "--markdown", action="store_true", help="print benchmarking results as a markdown table")
    return parser.parse_args()


def get_package_versions() -> Dict[str, str]:
    packages = [
        "albumentations",
        "imgaug",
        "torchvision",
        "tensorflow",
        "numpy",
        "opencv-python-headless",
        "scikit-image",
        "scipy",
        "pillow",
        "augmentor",
        "kornia",
        "augly",
    ]
    package_versions = {"Python": sys.version}
    for package in packages:
        with suppress(pkg_resources.DistributionNotFound):
            package_versions[package] = pkg_resources.get_distribution(package).version
    return package_versions


class BenchmarkTest:
    def __str__(self):
        return self.__class__.__name__

    def albumentations(self, img: np.ndarray) -> np.ndarray:
        img = self.albumentations_transform(img)
        return np.array(img, np.uint8, copy=False)

    def imgaug(self, img: np.ndarray) -> np.ndarray:
        img = self.imgaug_transform.augment_image(img)
        return np.array(img, np.uint8, copy=False)

    def augmentor(self, img: Image.Image) -> np.ndarray:
        img = self.augmentor_op.perform_operation([img])[0]
        return np.array(img, np.uint8, copy=False)

    def torchvision(self, img: torch.Tensor) -> np.ndarray:
        return self.torchvision_transform(img)

    def tensorflow(self, img: np.ndarray) -> np.ndarray:
        return self.tensorflow_transform(img)

    def kornia(self, img: torch.Tensor) -> np.ndarray:
        return self.kornia_transform(img)

    def augly(self, img: Image.Image) -> Image.Image:
        return self.augly_transform(img)

    def is_supported_by(self, library: str) -> bool:
        library_attr_map = {
            "imgaug": "imgaug_transform",
            "augmentor": ["augmentor_op", "augmentor_pipeline"],
            "kornia": "kornia_transform",
            "torchvision": "torchvision_transform",
            "albumentations": "albumentations_transform",
            "tensorflow": "tensorflow_transform",
            "augly": "augly_transform",
        }

        # Check if the library is in the map
        if library in library_attr_map:
            attrs = library_attr_map[library]
            # Ensure attrs is a list for uniform processing
            if not isinstance(attrs, list):
                attrs = [attrs]
            # Return True if any of the specified attributes exist
            return any(hasattr(self, attr) for attr in attrs)

        # Fallback: checks if the class has an attribute with the library's name
        return hasattr(self, library)

    def run(self, library: str, imgs: Union[List[np.ndarray], List[Image.Image], List[torch.Tensor]]) -> None:
        transform = getattr(self, library)
        for img in imgs:
            transform(img)


class HorizontalFlip(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Fliplr(p=1)
        self.augmentor_op = Operations.Flip(probability=1, top_bottom_left_right="LEFT_RIGHT")

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        return A.HorizontalFlip(p=1)(image=img)["image"]

    def torchvision_transform(self, img: np.ndarray) -> np.ndarray:
        return torchvision.hflip(img)

    def tensorflow_transform(self, img: np.ndarray) -> np.ndarray:
        return tf.image.flip_left_right(img)

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomHorizontalFlip(p=1)(img)

    def augly_transform(self, img: Image.Image) -> Image.Image:
        return imaugs.HFlip(p=1)(img)


class VerticalFlip(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Flipud(p=1)
        self.augmentor_op = Operations.Flip(probability=1, top_bottom_left_right="TOP_BOTTOM")

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        return A.VerticalFlip(p=1)(image=img)["image"]

    def torchvision_transform(self, img: np.ndarray) -> np.ndarray:
        return torchvision.vflip(img)

    def tensorflow_transform(self, img: np.ndarray) -> np.ndarray:
        return tf.image.flip_up_down(img)

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomVerticalFlip(p=1)(img)

    def augly_transform(self, img: Image.Image) -> Image.Image:
        return imaugs.VFlip(p=1)(img)


class Rotate(BenchmarkTest):
    def __init__(self):
        self.angle = 45
        self.imgaug_transform = iaa.Affine(rotate=(self.angle, self.angle), order=1, mode="reflect")
        self.augmentor_op = Operations.RotateStandard(probability=1, max_left_rotation=45, max_right_rotation=45)

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        return A.Rotate(limit=self.angle, p=1)(image=img)["image"]

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return T.rotate(img, angle=-self.angle, interpolation=InterpolationMode.BILINEAR)

    def tensorflow_transform(self, img: np.ndarray) -> np.ndarray:
        # Rotate the image by -45 degrees
        # Note: The 'rg' parameter specifies the rotation range in degrees. Here, we set it to 45 for both directions
        # to effectively rotate by -45 degrees. Adjust 'rg' as needed for your specific rotation requirements.
        return tf.keras.preprocessing.image.random_rotation(
            img,
            rg=self.angle,
            row_axis=1,
            col_axis=2,
            channel_axis=0,
            fill_mode="nearest",
            cval=0.0,
            interpolation_order=1,
        )

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        # Convert degrees to radians for rotation
        angle = torch.tensor(-self.angle) * (torch.pi / 180.0)

        # Perform rotation
        return K.geometry.transform.rotate(img, angle=angle, mode="bilinear", padding_mode="zeros")

    def augly_transform(self, img: Image.Image) -> Image.Image:
        return imaugs.RandomRotation(min_degrees=self.angle, max_degrees=self.angle, p=1)(img)


class BrightnessContrast(BenchmarkTest):
    def __init__(self):
        self.alpha = 1.5
        self.beta = 0.5
        self.imgaug_transform = iaa.Sequential(
            [
                iaa.Multiply((self.alpha, self.alpha), per_channel=False),
                iaa.Add((int(255 / self.beta), int(255 / self.beta)), per_channel=False),
            ]
        )

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        return A.RandomBrightnessContrast(p=1, contrast_limit=self.beta, brightness_limit=self.beta)(image=img)["image"]

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        img = torchvision.adjust_brightness(img, brightness_factor=self.alpha)
        return torchvision.adjust_contrast(img, contrast_factor=self.beta)

    def tensorflow_transform(self, img: tf.Tensor) -> tf.Tensor:
        img_bright = tf.image.adjust_brightness(img, delta=self.alpha)
        return tf.image.adjust_contrast(img_bright, contrast_factor=self.beta)

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        img_bright = K.enhance.adjust_brightness(img, self.alpha)
        return K.enhance.adjust_contrast(img_bright, self.beta)

    def augly_transform(self, img: Image.Image) -> Image.Image:
        return imaugs.Compose([imaugs.Brightness(factor=self.alpha, p=1), imaugs.Contrast(factor=self.beta)])(img)


class ShiftScaleRotate(BenchmarkTest):
    def __init__(self):
        self.angle = 25.0
        self.shift = (50, 50)
        self.scale = 2.0
        self.imgaug_transform = iaa.Affine(
            scale=(self.scale, self.scale),
            rotate=(self.angle, self.angle),
            translate_px=self.shift,
            order=1,
            mode="reflect",
        )

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        return A.ShiftScaleRotate(rotate_limit=self.angle, scale_limit=self.scale, shift_limit=self.shift, p=1)(
            image=img
        )["image"]

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return torchvision.affine(
            img,
            angle=self.angle,
            translate=self.shift,
            scale=self.scale,
            shear=0,
            interpolation=InterpolationMode.BILINEAR,
        )

    def tensorflow_transform(self, img: tf.Tensor) -> tf.Tensor:
        # Assuming 'img' is a numpy array of shape (height, width, channels)
        # Scale: 2x, Rotate: 25 degrees, Translate: 50 pixels in both x and y directions
        return tf.keras.preprocessing.image.apply_affine_transform(
            img,
            theta=self.angle,  # Rotation angle in degrees
            tx=self.shift[0],  # Translation in x
            ty=self.shift[0],  # Translation in y
            zx=self.scale,  # Zoom in x (scale)
            zy=self.scale,  # Zoom in y (scale), assuming you want uniform scaling
        )

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        # Rotate: 25 degrees, Scale: 2x (uniform scaling), Translate: 50 pixels in both x and y directions
        angle = torch.tensor([self.angle])  # Rotation angle in degrees
        scale = torch.tensor([[self.scale, self.scale]])  # Scaling factor
        translation = torch.tensor([list(self.shift)])  # Translation in pixels

        center = torch.tensor([img.shape[2] / 2, img.shape[1] / 2])[None, :]  # Add batch dimension with [None, :]

        # Create the 2D affine matrix for rotation + scaling + translation
        affine_matrix = K.geometry.transform.get_rotation_matrix2d(center, angle, scale)
        affine_matrix[..., 2] += translation  # Apply translation

        affine_matrix = affine_matrix.to(img.dtype)

        return K.geometry.transform.warp_affine(
            img, affine_matrix, dsize=(img.shape[3], img.shape[2]), mode="bilinear", padding_mode="zeros"
        )


class Equalize(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.AllChannelsHistogramEqualization()
        self.augmentor_op = Operations.HistogramEqualisation(probability=1)

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        return A.Equalize(p=1)(image=img)["image"]

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return torchvision.equalize(img)

    def tensorflow_transform(self, img: tf.Tensor) -> tf.Tensor:
        # Convert image to grayscale for histogram equalization
        img_gray = tf.image.rgb_to_grayscale(img)
        return tf.image.adjust_contrast(img_gray, 2)

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomEqualize(p=1)(img)


class RandomCrop64(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.CropToFixedSize(width=64, height=64)
        self.augmentor_op = Operations.Crop(probability=1, width=64, height=64, centre=False)

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        return A.RandomCrop(height=64, width=64, p=1)(image=img)["image"]

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return torchvision.crop(img, top=0, left=0, height=64, width=64)

    def tensorflow_transform(self, img: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(img)
        height, width = shape[:2]
        target_height, target_width = 64, 64

        # Conditionally resizing to ensure the image is at least 64x64
        img_resized = tf.cond(
            tf.logical_or(tf.less(height, target_height), tf.less(width, target_width)),
            lambda: tf.image.resize_with_pad(img, target_height, target_width),
            lambda: img,
        )

        # Randomly crop to 64x64
        return tf.image.random_crop(img_resized, size=[64, 64, 3])

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomCrop(size=(64, 64), p=1)(img)

    def augly_transform(self, img: Image.Image) -> Image.Image:
        x1 = 0.25
        x2 = random.uniform(0.25, 1)
        y1 = 0.25
        y2 = random.uniform(0.25, 1)
        return imaugs.Crop(x1=x1, y1=y1, x2=x2, y2=y2, p=1)(img)


class RandomSizedCrop_64_512(BenchmarkTest):
    def __init__(self):
        self.augmentor_pipeline = Pipeline()
        self.augmentor_pipeline.add_operation(Operations.Crop(probability=1, width=64, height=64, centre=False))
        self.augmentor_pipeline.add_operation(
            Operations.Resize(probability=1, width=512, height=512, resample_filter="BILINEAR")
        )
        self.imgaug_transform = iaa.Sequential(
            [iaa.CropToFixedSize(width=64, height=64), iaa.Resize(size=512, interpolation="linear")]
        )

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        img = A.random_crop(img, crop_height=64, crop_width=64, h_start=0, w_start=0)
        return A.resize(img, height=512, width=512)

    def augmentor(self, img: Image.Image) -> np.ndarray:
        for operation in self.augmentor_pipeline.operations:
            (img,) = operation.perform_operation([img])
        return img

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        img = torchvision.crop(img, top=0, left=0, height=64, width=64)
        return torchvision.resize(img, (512, 512))

    def tensorflow_transform(self, img: tf.Tensor) -> tf.Tensor:
        # Assume the input image size is at least 64x64. If not, resize to at least 64x64 before cropping.
        # This step ensures that random_crop can be applied safely.
        shape = tf.shape(img)
        min_dim = tf.reduce_min([shape[0], shape[1]])
        img_resized = tf.cond(
            tf.less(min_dim, 64),
            lambda: tf.image.resize_with_pad(img, 64, 64),  # Ensuring the image is at least 64x64
            lambda: img,
        )

        # Randomly crop to 64x64
        cropped_img = tf.image.random_crop(img_resized, size=[64, 64, 3])

        # Resize cropped image to 512x512
        return tf.image.resize(cropped_img, [512, 512], method=tf.image.ResizeMethod.BILINEAR)

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomResizedCrop(size=(512, 512), scale=(0.08, 1.0), ratio=(0.75, 1.33))(img)


class ShiftRGB(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Add((100, 100), per_channel=False)

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        return A.RGBShift(r_shift_limit=100, g_shift_limit=100, b_shift_limit=100, p=1)(image=img)["image"]

    def tensorflow_transform(self, img: tf.Tensor) -> tf.Tensor:
        # Create shifts for each channel
        r_shift = 100.0
        g_shift = 100.0
        b_shift = 100.0

        # Split the image tensor into its respective RGB channels
        r, g, b = tf.split(img, num_or_size_splits=3, axis=-1)

        # Apply the shifts
        r_shifted = r + r_shift
        g_shifted = g + g_shift
        b_shifted = b + b_shift

        # Clip the values to ensure they remain within [0, 255]
        r_clipped = tf.clip_by_value(r_shifted, clip_value_min=0, clip_value_max=255)
        g_clipped = tf.clip_by_value(g_shifted, clip_value_min=0, clip_value_max=255)
        b_clipped = tf.clip_by_value(b_shifted, clip_value_min=0, clip_value_max=255)

        # Concatenate the channels back together
        return tf.concat([r_clipped, g_clipped, b_clipped], axis=-1)

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomRGBShift(0.5, 0.5, 0.5, p=1)(img)  # Define the shifts for R, G, B channels


class PadToSize(BenchmarkTest):
    def __init__(self, target_height: int, target_width: int):
        self.target_height = target_height
        self.target_width = target_width

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        # Albumentations expects height first, then width
        transform = A.PadIfNeeded(
            min_height=self.target_height, min_width=self.target_width, border_mode=cv2.BORDER_REFLECT, p=1
        )
        return transform(image=img)["image"]

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        # Calculate padding
        height, width = img.shape[-2], img.shape[-1]
        pad_height = max(0, self.target_height - height)
        pad_width = max(0, self.target_width - width)

        # Symmetric padding
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # torchvision expects padding as [left, right, top, bottom]
        padding = [pad_left, pad_right, pad_top, pad_bottom]
        return torchvision.pad(img, padding, padding_mode="reflect")

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.PadTo(size=(self.target_width, self.target_height), pad_mode="reflect")(img)

    def tensorflow_transform(self, img: tf.Tensor) -> tf.Tensor:
        # Calculate padding sizes
        height_pad_needed = tf.maximum(self.target_height - tf.shape(img)[0], 0)
        width_pad_needed = tf.maximum(self.target_width - tf.shape(img)[1], 0)

        # Determine symmetric padding
        pad_top = height_pad_needed // 2
        pad_bottom = height_pad_needed - pad_top
        pad_left = width_pad_needed // 2
        pad_right = width_pad_needed - pad_left

        # Apply symmetric padding
        return tf.pad(img, paddings=[[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode="REFLECT")


class Resize512(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Resize(size=512, interpolation="linear")
        self.augmentor_op = Operations.Resize(probability=1, width=512, height=512, resample_filter="BILINEAR")

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        return A.resize(img, height=512, width=512)

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return torchvision.resize(img, (512, 512))

    def tensorflow_transform(self, img: tf.Tensor) -> tf.Tensor:
        return tf.image.resize(img, [512, 512], method="bilinear")

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return K.geometry.resize(img, (512, 512), interpolation="bilinear")

    def augly_transform(self, img: Image.Image) -> Image.Image:
        return imaugs.Resize(width=512, height=512, resample=Image.BILINEAR, p=1)(img)


class RandomGamma(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.GammaContrast(gamma=0.5)

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        return A.gamma_transform(img, gamma=0.5)

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return torchvision.adjust_gamma(img, gamma=0.5)

    def tensorflow_transform(self, img: tf.Tensor) -> tf.Tensor:
        # Normalize the image to [0, 1] by dividing by 255
        normalized_img = img / 255.0

        # Apply gamma correction with gamma = 0.5
        gamma_corrected_img = tf.pow(normalized_img, 0.5)

        # Rescale back to [0, 255] and convert to uint8
        return tf.clip_by_value(gamma_corrected_img * 255.0, 0, 255)

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomGamma(p=1.0, gamma=(0.5, 0.5))(img)


class Grayscale(BenchmarkTest):
    def __init__(self):
        self.augmentor_op = Operations.Greyscale(probability=1)
        self.imgaug_transform = iaa.Grayscale(alpha=1.0)

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        return A.to_gray(img)

    def augmentor(self, img: Image.Image) -> np.ndarray:
        img = self.augmentor_op.perform_operation([img])[0]
        img = np.array(img, np.uint8, copy=False)
        return np.dstack([img, img, img])

    def tensorflow_transform(self, img: tf.Tensor) -> tf.Tensor:
        return tf.image.rgb_to_grayscale(img)

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomGrayscale(p=1.0)(img)

    def augly_transform(self, img: Image.Image) -> Image.Image:
        return imaugs.Grayscale(p=1)(img)


class Multiply(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Multiply(mul=1.5)

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        return A.multiply(img, np.array([1.5]))

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        # Apply the multiplication
        multiplied_img = img * 1.5

        # Clip values to maintain them within [0, 1] range
        return torch.clamp(multiplied_img, min=0.0, max=255.0)

    def tensorflow_transform(self, img: tf.Tensor) -> tf.Tensor:
        img_normalized = img / 255.0 if tf.reduce_max(img) > 1.0 else img

        # Multiply pixel values by 1.5
        multiplied_img = img_normalized * 1.5

        # Clip values to maintain them within [0, 1] range
        multiplied_img_clipped = tf.clip_by_value(multiplied_img, clip_value_min=0.0, clip_value_max=1.0)

        # Convert back to [0, 255] range if necessary
        return multiplied_img_clipped * 255.0 if tf.reduce_max(img) > 1.0 else multiplied_img_clipped


class ColorJitter(BenchmarkTest):
    def __init__(self):
        self.brightness = 0.5
        self.contrast = 1.5
        self.saturation = 1.5
        self.hue = 0.5

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        return A.ColorJitter(
            brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue, p=1
        )(image=img)["image"]

    def torchvision(self, img: torch.Tensor) -> torch.Tensor:
        img = torchvision.adjust_brightness(img, self.brightness)
        img = torchvision.adjust_contrast(img, self.contrast)
        img = torchvision.adjust_saturation(img, self.saturation)
        return torchvision.adjust_hue(img, self.hue)

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.ColorJitter(
            brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue, p=1
        )(img)

    def augly_transform(self, img: Image.Image) -> Image.Image:
        return imaugs.ColorJitter(
            brightness_factor=self.brightness, contrast_factor=self.contrast, saturation_factor=self.saturation, p=1
        )(img)


class RandomPerspective(BenchmarkTest):
    def __init__(self):
        self.scale = [0.05, 0.1]

        self.imgaug_transform = iaa.PerspectiveTransform(scale=self.scale)

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        transform = A.Perspective(scale=self.scale, p=1)  # Adjust scale as needed
        return transform(image=img)["image"]

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomPerspective(distortion_scale=self.scale[0], p=1)(img)


class GaussianBlur(BenchmarkTest):
    def __init__(self, sigma: float = 2.0):
        self.sigma = sigma
        self.imgaug_transform = iaa.GaussianBlur(sigma=self.sigma)

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        transform = A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(self.sigma, self.sigma), p=1)
        return transform(image=img)["image"]

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return torchvision.gaussian_blur(img, kernel_size=[5, 5], sigma=self.sigma)

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomGaussianBlur(kernel_size=(5, 5), sigma=(self.sigma, self.sigma), p=1)(img)

    def augly_transform(self, img: Image.Image) -> Image.Image:
        return imaugs.Blur(radius=self.sigma, p=1)(img)


class MedianBlur(BenchmarkTest):
    def __init__(self, blur_limit: int = 5):
        # blur_limit or kernel size for median blur, ensuring it's an odd number
        self.blur_limit = blur_limit if blur_limit % 2 != 0 else blur_limit + 1
        self.imgaug_transform = iaa.MedianBlur(k=self.blur_limit)

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        transform = A.MedianBlur(blur_limit=(self.blur_limit, self.blur_limit), p=1)
        return transform(image=img)["image"]

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomMedianBlur(kernel_size=(self.blur_limit, self.blur_limit), p=1)(img)


class MotionBlur(BenchmarkTest):
    def __init__(self, kernel_size: int = 5, angle: float = 45, direction: float = 0.0):
        self.kernel_size = kernel_size  # Size of the motion blur kernel
        self.angle = angle  # Direction of the motion blur in degrees
        self.direction = direction  # Direction of the blur (used by ImgAug)
        self.imgaug_transform = iaa.MotionBlur(k=self.kernel_size, angle=[self.angle], direction=self.direction)

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        transform = A.MotionBlur(blur_limit=self.kernel_size, always_apply=True)
        return transform(image=img)["image"]

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomMotionBlur(kernel_size=self.kernel_size, angle=self.angle, direction=self.direction, p=1)(img)


class Posterize(BenchmarkTest):
    def __init__(self, bits: int = 4):
        self.bits = bits  # Number of bits to keep for each color channel
        self.imgaug_transform = iaa.Posterize(nb_bits=self.bits)

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        transform = A.Posterize(num_bits=self.bits, always_apply=True)
        return transform(image=img)["image"]

    def tensorflow_transform(self, img: tf.Tensor) -> tf.Tensor:
        # TensorFlow does not have a built-in posterize function, so we use a workaround
        # by scaling the image's values to the desired bit depth and back
        scale_factor = 255.0 / (2**self.bits - 1)
        img_scaled_down = tf.cast(img, tf.float32) / scale_factor
        img_scaled_down = tf.cast(img_scaled_down, tf.int32)
        return tf.cast(img_scaled_down, tf.float32) * scale_factor

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomPosterize(bits=self.bits, p=1)(img)


class JpegCompressionTransform(BenchmarkTest):
    def __init__(self, quality: int = 50):
        # Quality: Value between 0 and 100 (higher means better). In imgaug, it's 0-100 scale.
        self.quality = quality
        self.imgaug_transform = iaa.JpegCompression(compression=self.quality)

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        transform = A.ImageCompression(quality_lower=self.quality, quality_upper=self.quality, always_apply=True)
        return transform(image=img)["image"]

    def tensorflow_transform(self, img: tf.Tensor) -> tf.Tensor:
        # TensorFlow approach involves encoding and decoding the image.
        img_uint8 = tf.image.convert_image_dtype(img, tf.uint8)
        img_encoded = tf.image.encode_jpeg(img_uint8, format="", quality=self.quality)
        img_decoded = tf.image.decode_jpeg(img_encoded)
        return tf.image.convert_image_dtype(img_decoded, tf.float32)

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomJPEG(jpeg_quality=self.quality, p=1)(img)

    def augly_transform(self, img: Image.Image) -> Image.Image:
        return imaugs.EncodingQuality(quality=self.quality, p=1)(img)


class GaussianNoise(BenchmarkTest):
    def __init__(self, mean: float = 0, var: float = 0.010):
        self.mean = mean
        self.var = var
        self.imgaug_transform = iaa.AdditiveGaussianNoise(loc=self.mean, scale=(self.var**0.5, self.var**0.5))

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        transform = A.GaussNoise(var_limit=(20, 50), mean=self.mean, always_apply=True)
        return transform(image=img)["image"]

    def augly_transform(self, img: np.ndarray) -> np.ndarray:
        return imaugs.RandomNoise(mean=self.mean, std=self.var)(img)

    def tensorflow_transform(self, img: tf.Tensor) -> tf.Tensor:
        noise = tf.random.normal(shape=tf.shape(img), mean=self.mean, stddev=self.var**0.5)
        img_noisy = img + noise
        return tf.clip_by_value(img_noisy, 0.0, 255.0)

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        # Ensure img tensor is in float format; assume it is scaled between 0 and 1
        noise = torch.randn(img.size()) * self.var**0.5 + self.mean
        img_noisy = img + noise
        return torch.clamp(img_noisy, 0.0, 1.0)


def main() -> None:
    args = parse_args()
    package_versions = get_package_versions()
    if args.print_package_versions:
        pass

    images_per_second = defaultdict(dict)
    libraries = args.libraries
    data_dir = Path(args.data_dir)
    paths = sorted(data_dir.glob("*.*"))
    paths = paths[: args.images]
    imgs_cv2 = [read_img_cv2(path) for path in paths]
    imgs_pillow = [read_img_pillow(path) for path in paths]
    imgs_torch = [read_img_torch(path) for path in paths]
    imgs_tensorflow = [read_img_tensorflow(path) for path in paths]
    imgs_kornia = [read_img_kornia(path) for path in paths]

    benchmarks = [
        HorizontalFlip(),
        VerticalFlip(),
        Rotate(),
        BrightnessContrast(),
        ShiftScaleRotate(),
        Equalize(),
        RandomCrop64(),
        RandomGamma(),
        Grayscale(),
        PadToSize(1024, 1024),
        Resize512(),
        RandomSizedCrop_64_512(),
        ShiftRGB(),
        Multiply(),
        ColorJitter(),
        RandomPerspective(),
        GaussianBlur(),
        MedianBlur(),
        MotionBlur(),
        Posterize(),
    ]
    for library in libraries:
        if library in ("augmentor", "augly"):
            imgs = imgs_pillow
        elif library == "torchvision":
            imgs = imgs_torch
        elif library == "tensorflow":
            imgs = imgs_tensorflow
        elif library == "kornia":
            imgs = imgs_kornia
        else:
            imgs = imgs_cv2

        pbar = tqdm(total=len(benchmarks))
        for benchmark in benchmarks:
            pbar.set_description(f"Current benchmark: {library} | {benchmark}")
            benchmark_images_per_second = None
            if benchmark.is_supported_by(library):
                timer = Timer(lambda: benchmark.run(library, imgs))
                run_times = timer.repeat(number=1, repeat=args.runs)
                benchmark_images_per_second = [1 / (run_time / args.images) for run_time in run_times]
            images_per_second[library][str(benchmark)] = benchmark_images_per_second
            pbar.update(1)
        pbar.close()
    pd.set_option("display.width", 1000)
    df = pd.DataFrame.from_dict(images_per_second)
    df = df.map(lambda r: format_results(r, args.show_std))
    df = df[libraries]
    augmentations = [str(i) for i in benchmarks]
    df = df.reindex(augmentations)
    if args.markdown:
        makedown_generator = MarkdownGenerator(df, package_versions)
        makedown_generator.print()
    else:
        pass


if __name__ == "__main__":
    main()
