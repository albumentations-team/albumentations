import argparse
import math
import os
import sys
from abc import ABC
from collections import defaultdict
from contextlib import suppress
from timeit import Timer

import cv2
import numpy as np
import pandas as pd
import pkg_resources
import solt.core as slc
import solt.transforms as slt
import tensorflow as tf
import torchvision.transforms.functional as torchvision
from Augmentor import Operations, Pipeline
from imgaug import augmenters as iaa
from PIL import Image, ImageOps
from pytablewriter import MarkdownTableWriter
from pytablewriter.style import Style
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as T
from tqdm import tqdm

import albumentations as A

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


DEFAULT_BENCHMARKING_LIBRARIES = ["albumentations", "imgaug", "torchvision", "keras", "augmentor", "solt"]


def parse_args():
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


def get_package_versions():
    packages = [
        "albumentations",
        "imgaug",
        "torchvision",
        "keras",
        "numpy",
        "opencv-python",
        "scikit-image",
        "scipy",
        "pillow",
        "augmentor",
        "solt",
    ]
    package_versions = {"Python": sys.version}
    for package in packages:
        with suppress(pkg_resources.DistributionNotFound):
            package_versions[package] = pkg_resources.get_distribution(package).version
    return package_versions


class MarkdownGenerator:
    def __init__(self, df, package_versions):
        self._df = df
        self._package_versions = package_versions

    def _highlight_best_result(self, results):
        best_result = float("-inf")
        for result in results:
            try:
                result = int(result)
            except ValueError:
                continue
            if result > best_result:
                best_result = result
        return [f"**{r}**" if r == str(best_result) else r for r in results]

    def _make_headers(self):
        libraries = self._df.columns.to_list()
        columns = []
        for library in libraries:
            version = self._package_versions[library]

            columns.append(f"{library}<br><small>{version}</small>")
        return ["", *columns]

    def _make_value_matrix(self):
        index = self._df.index.tolist()
        values = self._df.values.tolist()
        value_matrix = []
        for transform, results in zip(index, values):
            row = [transform, *self._highlight_best_result(results)]
            value_matrix.append(row)
        return value_matrix

    def _make_versions_text(self):
        libraries = ["Python", "numpy", "pillow", "opencv-python", "scikit-image", "scipy"]
        libraries_with_versions = [
            "{library} {version}".format(library=library, version=self._package_versions[library].replace("\n", ""))
            for library in libraries
        ]
        return "Python and library versions: {}.".format(", ".join(libraries_with_versions))

    def print(self) -> None:
        writer = MarkdownTableWriter()
        writer.headers = self._make_headers()
        writer.value_matrix = self._make_value_matrix()
        writer.styles = [Style(align="left")] + [Style(align="center") for _ in range(len(writer.headers) - 1)]
        writer.write_table()


def read_img_pillow(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def read_img_cv2(filepath):
    img = cv2.imread(filepath)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def format_results(images_per_second_for_aug, show_std=False):
    if images_per_second_for_aug is None:
        return "-"
    result = str(math.floor(np.mean(images_per_second_for_aug)))
    if show_std:
        result += f" ± {math.ceil(np.std(images_per_second_for_aug))}"
    return result


class BenchmarkTest(ABC):
    def __str__(self):
        return self.__class__.__name__

    def imgaug(self, img):
        return self.imgaug_transform.augment_image(img)

    def augmentor(self, img):
        img = self.augmentor_op.perform_operation([img])[0]
        return np.array(img, np.uint8, copy=False)

    def solt(self, img):
        return self.solt_stream({"image": img}, return_torch=False).data[0]

    def torchvision(self, img):
        img = self.torchvision_transform(img)
        return np.array(img, np.uint8, copy=False)

    def is_supported_by(self, library):
        if library == "imgaug":
            return hasattr(self, "imgaug_transform")
        if library == "augmentor":
            return hasattr(self, "augmentor_op") or hasattr(self, "augmentor_pipeline")
        if library == "solt":
            return hasattr(self, "solt_stream")
        if library == "torchvision":
            return hasattr(self, "torchvision_transform")

        return hasattr(self, library)

    def run(self, library, imgs) -> None:
        transform = getattr(self, library)
        for img in imgs:
            transform(img)


class HorizontalFlip(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Fliplr(p=1)
        self.augmentor_op = Operations.Flip(probability=1, top_bottom_left_right="LEFT_RIGHT")
        self.solt_stream = slc.Stream([slt.Flip(p=1, axis=1)])

    def albumentations(self, img):
        return A.hflip_cv2(img)

    def torchvision_transform(self, img):
        return torchvision.hflip(img)

    def keras(self, img: np.ndarray):
        return tf.image.flip_left_right(img).numpy()

    def imgaug(self, img):
        return np.ascontiguousarray(self.imgaug_transform.augment_image(img))


class VerticalFlip(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Flipud(p=1)
        self.augmentor_op = Operations.Flip(probability=1, top_bottom_left_right="TOP_BOTTOM")
        self.solt_stream = slc.Stream([slt.Flip(p=1, axis=0)])

    def albumentations(self, img: np.ndarray) -> np.ndarray:
        return A.vflip(img)

    def torchvision_transform(self, img: np.ndarray) -> np.ndarray:
        return torchvision.vflip(img)

    def keras(self, img: np.ndarray) -> np.ndarray:
        return tf.image.flip_up_down(img).numpy()

    def imgaug(self, img: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(self.imgaug_transform.augment_image(img))


class Rotate(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Affine(rotate=(45, 45), order=1, mode="reflect")
        self.augmentor_op = Operations.RotateStandard(probability=1, max_left_rotation=45, max_right_rotation=45)
        self.solt_stream = slc.Stream([slt.Rotate(p=1, angle_range=(45, 45))], padding="r")

    def albumentations(self, img: np.ndarray) -> np.ndarray:
        return A.rotate(img, angle=-45)

    def torchvision_transform(self, img):
        return T.rotate(img, angle=-45, interpolation=InterpolationMode.BILINEAR)

    def keras(self, img):
        # Rotate the image by -45 degrees
        # Note: The 'rg' parameter specifies the rotation range in degrees. Here, we set it to 45 for both directions
        # to effectively rotate by -45 degrees. Adjust 'rg' as needed for your specific rotation requirements.
        return tf.keras.preprocessing.image.random_rotation(
            img, rg=45, row_axis=1, col_axis=2, channel_axis=0, fill_mode="nearest", cval=0.0, interpolation_order=1
        )


class BrightnessContrast(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Sequential(
            [iaa.Multiply((1.5, 1.5), per_channel=False), iaa.Add((127, 127), per_channel=False)]
        )
        self.augmentor_pipeline = Pipeline()
        self.augmentor_pipeline.add_operation(
            Operations.RandomBrightness(probability=1, min_factor=1.5, max_factor=1.5)
        )
        self.augmentor_pipeline.add_operation(Operations.RandomContrast(probability=1, min_factor=1.5, max_factor=1.5))
        self.solt_stream = slc.Stream(
            [slt.Brightness(p=1, brightness_range=(127, 127)), slt.Contrast(p=1, contrast_range=(1.5, 1.5))]
        )

    def albumentations(self, img):
        return A.brightness_contrast_adjust(img, alpha=1.5, beta=0.5, beta_by_max=True)

    def torchvision_transform(self, img):
        img = torchvision.adjust_brightness(img, brightness_factor=1.5)
        return torchvision.adjust_contrast(img, contrast_factor=1.5)

    def augmentor(self, img):
        for operation in self.augmentor_pipeline.operations:
            (img,) = operation.perform_operation([img])
        return np.array(img, np.uint8, copy=False)

    def keras(self, img):
        # Convert to tensor
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        # Adjust brightness by adding delta
        img_bright = tf.image.adjust_brightness(img_tensor, delta=0.1)
        # Adjust contrast
        img_contrast = tf.image.adjust_contrast(img_bright, contrast_factor=1.5)
        # Convert back to numpy array
        return img_contrast.numpy()


class ShiftScaleRotate(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Affine(
            scale=(2, 2), rotate=(25, 25), translate_px=(50, 50), order=1, mode="reflect"
        )

    def albumentations(self, img):
        return A.shift_scale_rotate(img, angle=-45, scale=2, dx=0.2, dy=0.2)

    def torchvision_transform(self, img):
        return torchvision.affine(
            img, angle=25, translate=(50, 50), scale=2, shear=0, interpolation=InterpolationMode.BILINEAR
        )


class ShiftHSV(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.AddToHueAndSaturation((20, 20), per_channel=False)
        self.solt_stream = slc.Stream([slt.HSV(p=1, h_range=(20, 20), s_range=(20, 20), v_range=(20, 20))])

    def albumentations(self, img):
        return A.shift_hsv(img, hue_shift=20, sat_shift=20, val_shift=20)

    def torchvision_transform(self, img):
        img = torchvision.adjust_hue(img, hue_factor=0.1)
        img = torchvision.adjust_saturation(img, saturation_factor=1.2)
        return torchvision.adjust_brightness(img, brightness_factor=1.2)

    def keras(self, img):
        # Ensure img is a tensor in the range [0, 255]
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

        # Normalize the image to [0, 1] by dividing by 255
        normalized_img = img_tensor / 255.0

        # Convert RGB to HSV
        hsv_img = tf.image.rgb_to_hsv(normalized_img)

        # Shift Hue, Saturation, and Value (Brightness)
        # Hue values are in [0, 1], so a shift of 20 degrees would be (20/360) in the HSV color space.
        hue_shift = 20 / 360.0
        sat_shift = 0.2  # Assuming a 20% increase in saturation
        value_shift = 0.2  # Assuming a 20% increase in brightness

        # Apply shifts
        hsv_img = tf.stack(
            [
                tf.clip_by_value(hsv_img[..., 0] + hue_shift, 0, 1),
                tf.clip_by_value(hsv_img[..., 1] * (1 + sat_shift), 0, 1),
                tf.clip_by_value(hsv_img[..., 2] * (1 + value_shift), 0, 1),
            ],
            axis=-1,
        )

        # Convert back to RGB
        rgb_img = tf.image.hsv_to_rgb(hsv_img)

        # Rescale back to [0, 255] and convert to uint8
        final_img = tf.clip_by_value(rgb_img * 255.0, 0, 255)
        final_img = tf.cast(final_img, tf.uint8)


class Equalize(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.AllChannelsHistogramEqualization()
        self.augmentor_op = Operations.HistogramEqualisation(probability=1)

    def albumentations(self, img):
        return A.equalize(img)

    def imgaug(self, img):
        img = self.imgaug_transform.augment_image(img)
        return np.ascontiguousarray(img)


class RandomCrop64(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.CropToFixedSize(width=64, height=64)
        self.augmentor_op = Operations.Crop(probability=1, width=64, height=64, centre=False)
        self.solt_stream = slc.Stream([slt.Crop(crop_to=(64, 64), crop_mode="r")])

    def albumentations(self, img):
        img = A.random_crop(img, crop_height=64, crop_width=64, h_start=0, w_start=0)
        return np.ascontiguousarray(img)

    def torchvision_transform(self, img):
        return torchvision.crop(img, top=0, left=0, height=64, width=64)

    def solt(self, img):
        img = self.solt_stream({"image": img}, return_torch=False).data[0]
        return np.ascontiguousarray(img)

    def keras(self, img):
        # Ensure img is a tensor in the range [0, 255] and has rank 3
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

        # TensorFlow's random_crop expects the image tensor's shape to be known,
        # so ensure your image tensor has a fully defined shape.
        # We also need the image to be at least 64x64 pixels.
        # Check and potentially resize the image to meet the requirements.
        shape = tf.shape(img_tensor)
        height, width = shape[0], shape[1]
        target_height, target_width = 64, 64

        # Conditionally resizing to ensure the image is at least 64x64
        img_resized = tf.cond(
            tf.logical_or(tf.less(height, target_height), tf.less(width, target_width)),
            lambda: tf.image.resize_with_pad(img_tensor, target_height, target_width),
            lambda: img_tensor,
        )

        # Randomly crop to 64x64
        cropped_img = tf.image.random_crop(img_resized, size=[64, 64, 3])

        # Convert back to numpy array
        return cropped_img.numpy()


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
        self.solt_stream = slc.Stream([slt.Crop(crop_to=(64, 64), crop_mode="r"), slt.Resize(resize_to=(512, 512))])

    def albumentations(self, img):
        img = A.random_crop(img, crop_height=64, crop_width=64, h_start=0, w_start=0)
        return A.resize(img, height=512, width=512)

    def augmentor(self, img):
        for operation in self.augmentor_pipeline.operations:
            (img,) = operation.perform_operation([img])
        return np.array(img, np.uint8, copy=False)

    def torchvision_transform(self, img):
        img = torchvision.crop(img, top=0, left=0, height=64, width=64)
        return torchvision.resize(img, (512, 512))

    def keras(self, img):
        # Ensure img is a tensor in the range [0, 255]
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

        # Assume the input image size is at least 64x64. If not, resize to at least 64x64 before cropping.
        # This step ensures that random_crop can be applied safely.
        shape = tf.shape(img_tensor)
        min_dim = tf.reduce_min([shape[0], shape[1]])
        img_resized = tf.cond(
            tf.less(min_dim, 64),
            lambda: tf.image.resize_with_pad(img_tensor, 64, 64),  # Ensuring the image is at least 64x64
            lambda: img_tensor,
        )

        # Randomly crop to 64x64
        cropped_img = tf.image.random_crop(img_resized, size=[64, 64, 3])

        # Resize cropped image to 512x512
        resized_img = tf.image.resize(cropped_img, [512, 512], method=tf.image.ResizeMethod.BILINEAR)

        # Convert back to numpy array, assuming the original image was uint8
        final_img = tf.cast(resized_img, tf.uint8)

        return final_img.numpy()


class ShiftRGB(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Add((100, 100), per_channel=False)

    def albumentations(self, img):
        return A.shift_rgb(img, r_shift=100, g_shift=100, b_shift=100)

    def keras(self, img):
        # Ensure img is a tensor in the range [0, 255]
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

        # Create shifts for each channel
        r_shift = 100.0
        g_shift = 100.0
        b_shift = 100.0

        # Split the image tensor into its respective RGB channels
        r, g, b = tf.split(img_tensor, num_or_size_splits=3, axis=-1)

        # Apply the shifts
        r_shifted = r + r_shift
        g_shifted = g + g_shift
        b_shifted = b + b_shift

        # Clip the values to ensure they remain within [0, 255]
        r_clipped = tf.clip_by_value(r_shifted, clip_value_min=0, clip_value_max=255)
        g_clipped = tf.clip_by_value(g_shifted, clip_value_min=0, clip_value_max=255)
        b_clipped = tf.clip_by_value(b_shifted, clip_value_min=0, clip_value_max=255)

        # Concatenate the channels back together
        shifted_img = tf.concat([r_clipped, g_clipped, b_clipped], axis=-1)

        # Convert back to numpy array
        final_img = tf.cast(shifted_img, tf.uint8).numpy()

        return final_img


class PadToSize512(BenchmarkTest):
    def __init__(self):
        self.solt_stream = slc.Stream([slt.Pad(pad_to=(512, 512), padding="r")])

    def albumentations(self, img):
        return A.pad(img, min_height=512, min_width=512)

    def torchvision_transform(self, img):
        if img.size[0] < 512:
            img = torchvision.pad(img, (int((1 + 512 - img.size[0]) / 2), 0), padding_mode="reflect")
        if img.size[1] < 512:
            img = torchvision.pad(img, (0, int((1 + 512 - img.size[1]) / 2)), padding_mode="reflect")
        return img


class Resize512(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Resize(size=512, interpolation="linear")
        self.solt_stream = slc.Stream([slt.Resize(resize_to=(512, 512))])
        self.augmentor_op = Operations.Resize(probability=1, width=512, height=512, resample_filter="BILINEAR")

    def albumentations(self, img):
        return A.resize(img, height=512, width=512)

    def torchvision_transform(self, img):
        return torchvision.resize(img, (512, 512))

    def keras(self, img):
        # Ensure img is a tensor
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

        # Resize the image to 512x512 pixels
        # tf.image.resize expects the size in the format [height, width]
        resized_img = tf.image.resize(img_tensor, [512, 512], method="bilinear")

        # Convert back to numpy array and adjust dtype if necessary
        # Assuming the original dtype was uint8, we cast the resized image back to uint8
        return tf.cast(resized_img, tf.uint8).numpy()


class RandomGamma(BenchmarkTest):
    def __init__(self):
        self.solt_stream = slc.Stream([slt.GammaCorrection(p=1, gamma_range=(0.5, 0.5))])

    def albumentations(self, img):
        return A.gamma_transform(img, gamma=0.5)

    def torchvision_transform(self, img):
        return torchvision.adjust_gamma(img, gamma=0.5)

    def keras(self, img):
        # Ensure img is a tensor in the range [0, 255]
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

        # Normalize the image to [0, 1] by dividing by 255
        normalized_img = img_tensor / 255.0

        # Apply gamma correction with gamma = 0.5
        gamma_corrected_img = tf.pow(normalized_img, 0.5)

        # Rescale back to [0, 255] and convert to uint8
        rescaled_img = tf.clip_by_value(gamma_corrected_img * 255.0, 0, 255)
        final_img = tf.cast(rescaled_img, tf.uint8)

        # Convert back to numpy array
        return final_img.numpy()


class Grayscale(BenchmarkTest):
    def __init__(self):
        self.augmentor_op = Operations.Greyscale(probability=1)
        self.imgaug_transform = iaa.Grayscale(alpha=1.0)
        self.solt_stream = slc.Stream([slt.CvtColor(mode="rgb2gs")])

    def albumentations(self, img):
        return A.to_gray(img)

    def torchvision_transform(self, img):
        return torchvision.to_grayscale(img, num_output_channels=3)

    def augmentor(self, img):
        img = self.augmentor_op.perform_operation([img])[0]
        img = np.array(img, np.uint8, copy=False)
        return np.dstack([img, img, img])

    def keras(self, img):
        # Convert to tensor
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        # Convert to grayscale
        return tf.image.rgb_to_grayscale(img_tensor).numpy()


class Multiply(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Multiply(mul=1.5)

    def albumentations(self, img):
        return A.multiply(img, np.array([1.5]))


class MultiplyElementwise(BenchmarkTest):
    def __init__(self):
        self.aug = A.MultiplicativeNoise((1.5, 1.5), per_channel=True, elementwise=True, p=1)
        self.imgaug_transform = iaa.MultiplyElementwise(mul=(1.5, 1.5), per_channel=True)

    def albumentations(self, img):
        return self.aug(image=img)["image"]

    def keras(self, img):
        # Ensure img is a tensor
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

        # Multiply the image tensor by 1.5
        # Note: This might increase pixel values beyond the typical range for uint8 images ([0, 255]).
        # You may want to clip the values to stay within this range depending on your application's needs.
        multiplied_img = img_tensor * 1.5

        # Clip values to [0, 255] and cast to uint8 to maintain image data format
        clipped_img = tf.clip_by_value(multiplied_img, clip_value_min=0, clip_value_max=255)
        final_img = tf.cast(clipped_img, tf.uint8)

        # Convert back to numpy array
        return final_img.numpy()


class ColorJitter(BenchmarkTest):
    def __init__(self):
        imgaug_hue_param = int(0.5 * 255)
        self.imgaug_transform = iaa.AddToHue((imgaug_hue_param, imgaug_hue_param))

    def imgaug(self, img):
        img = iaa.pillike.enhance_brightness(img, 1.5)
        img = iaa.pillike.enhance_contrast(img, 1.5)
        img = iaa.pillike.enhance_color(img, 1.5)
        return self.imgaug_transform.augment_image(img)

    def albumentations(self, img):
        img = A.adjust_brightness_torchvision(img, 1.5)
        img = A.adjust_contrast_torchvision(img, 1.5)
        img = A.adjust_saturation_torchvision(img, 1.5)
        return A.adjust_hue_torchvision(img, 0.5)

    def torchvision_transform(self, img):
        img = torchvision.adjust_brightness(img, 1.5)
        img = torchvision.adjust_contrast(img, 1.5)
        img = torchvision.adjust_saturation(img, 1.5)
        return torchvision.adjust_hue(img, 0.5)

    def keras(self, img):
        # Ensure img is a float32 tensor in the range [0, 1]
        img = tf.convert_to_tensor(img, dtype=tf.float32) / 255.0

        # Adjust brightness (+50%)
        img_bright = tf.image.adjust_brightness(img, delta=0.5)

        # Adjust contrast (*1.5)
        img_contrast = tf.image.adjust_contrast(img_bright, contrast_factor=1.5)

        # Adjust hue (+0.1 of the 0-1 range, equivalent to +36 degrees since 0.1 * 360 = 36)
        img_hue = tf.image.adjust_hue(img_contrast, delta=0.1)

        # Adjust saturation (*1.5)
        # This is more complex in TensorFlow as it doesn't have a direct saturation function
        # Convert RGB to HSV, scale the S channel, convert back to RGB
        img_hsv = tf.image.rgb_to_hsv(img_hue)
        hue, saturation, value = tf.split(img_hsv, 3, axis=-1)
        saturation = saturation * 1.5
        # Ensure saturation remains within [0, 1]
        saturation = tf.clip_by_value(saturation, clip_value_min=0, clip_value_max=1)
        img_adjusted_hsv = tf.concat([hue, saturation, value], axis=-1)
        img_saturation = tf.image.hsv_to_rgb(img_adjusted_hsv)

        # Convert back to numpy array and scale to [0, 255]
        return (img_saturation.numpy() * 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    package_versions = get_package_versions()
    if args.print_package_versions:
        pass
    images_per_second = defaultdict(dict)
    libraries = args.libraries
    data_dir = args.data_dir
    paths = sorted(os.listdir(data_dir))
    paths = paths[: args.images]
    imgs_cv2 = [read_img_cv2(os.path.join(data_dir, path)) for path in paths]
    imgs_pillow = [read_img_pillow(os.path.join(data_dir, path)) for path in paths]

    benchmarks = [
        HorizontalFlip(),
        VerticalFlip(),
        Rotate(),
        ShiftScaleRotate(),
        BrightnessContrast(),
        ShiftRGB(),
        ShiftHSV(),
        RandomGamma(),
        Grayscale(),
        RandomCrop64(),
        PadToSize512(),
        Resize512(),
        RandomSizedCrop_64_512(),
        Equalize(),
        Multiply(),
        MultiplyElementwise(),
        ColorJitter(),
    ]
    for library in libraries:
        imgs = imgs_pillow if library in ("torchvision", "augmentor") else imgs_cv2
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
