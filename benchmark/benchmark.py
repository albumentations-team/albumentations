from __future__ import division, print_function
import argparse
import math
import os
import sys
from abc import ABC
from timeit import Timer
from collections import defaultdict
import pkg_resources

from Augmentor import Operations, Pipeline
from PIL import Image, ImageOps
from pytablewriter import MarkdownTableWriter
from pytablewriter.style import Style

import cv2

cv2.setNumThreads(0)  # noqa E402
cv2.ocl.setUseOpenCL(False)  # noqa E402

os.environ["OMP_NUM_THREADS"] = "1"  # noqa E402
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # noqa E402
os.environ["MKL_NUM_THREADS"] = "1"  # noqa E402
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # noqa E402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa E402

from tqdm import tqdm
import numpy as np
import pandas as pd
import torchvision.transforms.functional as torchvision
import keras_preprocessing.image as keras
from imgaug import augmenters as iaa
import solt.core as slc
import solt.transforms as slt
import solt.data as sld

import albumentations.augmentations.functional as albumentations
import albumentations as A


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
        "pillow-simd",
        "augmentor",
        "solt",
    ]
    package_versions = {"Python": sys.version}
    for package in packages:
        try:
            package_versions[package] = pkg_resources.get_distribution(package).version
        except pkg_resources.DistributionNotFound:
            pass
    return package_versions


class MarkdownGenerator:
    def __init__(self, df, package_versions):
        self._df = df
        self._package_versions = package_versions
        self._libraries_description = {"torchvision": "(Pillow-SIMD backend)"}

    def _highlight_best_result(self, results):
        best_result = float("-inf")
        for result in results:
            try:
                result = int(result)
            except ValueError:
                continue
            if result > best_result:
                best_result = result
        return ["**{}**".format(r) if r == str(best_result) else r for r in results]

    def _make_headers(self):
        libraries = self._df.columns.to_list()
        columns = []
        for library in libraries:
            version = self._package_versions[library]
            library_description = self._libraries_description.get(library)
            if library_description:
                library += " {}".format(library_description)

            columns.append("{library}<br><small>{version}</small>".format(library=library, version=version))
        return [""] + columns

    def _make_value_matrix(self):
        index = self._df.index.tolist()
        values = self._df.values.tolist()
        value_matrix = []
        for transform, results in zip(index, values):
            row = [transform] + self._highlight_best_result(results)
            value_matrix.append(row)
        return value_matrix

    def _make_versions_text(self):
        libraries = ["Python", "numpy", "pillow-simd", "opencv-python", "scikit-image", "scipy"]
        libraries_with_versions = [
            "{library} {version}".format(library=library, version=self._package_versions[library].replace("\n", ""))
            for library in libraries
        ]
        return "Python and library versions: {}.".format(", ".join(libraries_with_versions))

    def print(self):
        writer = MarkdownTableWriter()
        writer.headers = self._make_headers()
        writer.value_matrix = self._make_value_matrix()
        writer.styles = [Style(align="left")] + [Style(align="center") for _ in range(len(writer.headers) - 1)]
        writer.write_table()
        print("\n" + self._make_versions_text())


def read_img_pillow(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def read_img_cv2(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def format_results(images_per_second_for_aug, show_std=False):
    if images_per_second_for_aug is None:
        return "-"
    result = str(math.floor(np.mean(images_per_second_for_aug)))
    if show_std:
        result += " ± {}".format(math.ceil(np.std(images_per_second_for_aug)))
    return result


class BenchmarkTest(ABC):
    def __str__(self):
        return self.__class__.__name__

    def imgaug(self, img):
        return self.imgaug_transform.augment_image(img)

    def augmentor(self, img):
        img = self.augmentor_op.perform_operation([img])[0]
        return np.array(img, np.uint8, copy=True)

    def solt(self, img):
        dc = sld.DataContainer(img, "I")
        dc = self.solt_stream(dc)
        return dc.data[0]

    def torchvision(self, img):
        img = self.torchvision_transform(img)
        return np.array(img, np.uint8, copy=True)

    def is_supported_by(self, library):
        if library == "imgaug":
            return hasattr(self, "imgaug_transform")
        elif library == "augmentor":
            return hasattr(self, "augmentor_op") or hasattr(self, "augmentor_pipeline")
        elif library == "solt":
            return hasattr(self, "solt_stream")
        elif library == "torchvision":
            return hasattr(self, "torchvision_transform")
        else:
            return hasattr(self, library)

    def run(self, library, imgs):
        transform = getattr(self, library)
        for img in imgs:
            transform(img)


class HorizontalFlip(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Fliplr(p=1)
        self.augmentor_op = Operations.Flip(probability=1, top_bottom_left_right="LEFT_RIGHT")
        self.solt_stream = slc.Stream([slt.RandomFlip(p=1, axis=1)])

    def albumentations(self, img):
        if img.ndim == 3 and img.shape[2] > 1 and img.dtype == np.uint8:
            return albumentations.hflip_cv2(img)
        else:
            return albumentations.hflip(img)

    def torchvision_transform(self, img):
        return torchvision.hflip(img)

    def keras(self, img):
        return np.ascontiguousarray(keras.flip_axis(img, axis=1))

    def imgaug(self, img):
        return np.ascontiguousarray(self.imgaug_transform.augment_image(img))


class VerticalFlip(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Flipud(p=1)
        self.augmentor_op = Operations.Flip(probability=1, top_bottom_left_right="TOP_BOTTOM")
        self.solt_stream = slc.Stream([slt.RandomFlip(p=1, axis=0)])

    def albumentations(self, img):
        return albumentations.vflip(img)

    def torchvision_transform(self, img):
        return torchvision.vflip(img)

    def keras(self, img):
        return np.ascontiguousarray(keras.flip_axis(img, axis=0))

    def imgaug(self, img):
        return np.ascontiguousarray(self.imgaug_transform.augment_image(img))


class Rotate(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Affine(rotate=(45, 45), order=1, mode="reflect")
        self.augmentor_op = Operations.RotateStandard(probability=1, max_left_rotation=45, max_right_rotation=45)
        self.solt_stream = slc.Stream([slt.RandomRotate(p=1, rotation_range=(45, 45))], padding="r")

    def albumentations(self, img):
        return albumentations.rotate(img, angle=-45)

    def torchvision_transform(self, img):
        return torchvision.rotate(img, angle=-45, resample=Image.BILINEAR)

    def keras(self, img):
        img = keras.apply_affine_transform(img, theta=45, channel_axis=2, fill_mode="reflect")
        return np.ascontiguousarray(img)


class Brightness(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Add((127, 127), per_channel=False)
        self.augmentor_op = Operations.RandomBrightness(probability=1, min_factor=1.5, max_factor=1.5)
        self.solt_stream = slc.Stream([slt.ImageRandomBrightness(p=1, brightness_range=(127, 127))])

    def albumentations(self, img):
        return albumentations.brightness_contrast_adjust(img, beta=0.5, beta_by_max=True)

    def torchvision_transform(self, img):
        return torchvision.adjust_brightness(img, brightness_factor=1.5)

    def keras(self, img):
        return keras.apply_brightness_shift(img, brightness=1.5).astype(np.uint8)


class Contrast(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Multiply((1.5, 1.5), per_channel=False)
        self.augmentor_op = Operations.RandomContrast(probability=1, min_factor=1.5, max_factor=1.5)
        self.solt_stream = slc.Stream([slt.ImageRandomContrast(p=1, contrast_range=(1.5, 1.5))])

    def albumentations(self, img):
        return albumentations.brightness_contrast_adjust(img, alpha=1.5)

    def torchvision_transform(self, img):
        return torchvision.adjust_contrast(img, contrast_factor=1.5)


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
            [
                slt.ImageRandomBrightness(p=1, brightness_range=(127, 127)),
                slt.ImageRandomContrast(p=1, contrast_range=(1.5, 1.5)),
            ]
        )

    def albumentations(self, img):
        return albumentations.brightness_contrast_adjust(img, alpha=1.5, beta=0.5, beta_by_max=True)

    def torchvision_transform(self, img):
        img = torchvision.adjust_brightness(img, brightness_factor=1.5)
        img = torchvision.adjust_contrast(img, contrast_factor=1.5)
        return img

    def augmentor(self, img):
        for operation in self.augmentor_pipeline.operations:
            img, = operation.perform_operation([img])
        return np.array(img, np.uint8, copy=True)


class ShiftScaleRotate(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Affine(
            scale=(2, 2), rotate=(45, 45), translate_px=(50, 50), order=1, mode="reflect"
        )

    def albumentations(self, img):
        return albumentations.shift_scale_rotate(img, angle=-45, scale=2, dx=0.2, dy=0.2)

    def torchvision_transform(self, img):
        return torchvision.affine(img, angle=45, translate=(50, 50), scale=2, shear=0, resample=Image.BILINEAR)

    def keras(self, img):
        img = keras.apply_affine_transform(img, theta=45, tx=50, ty=50, zx=0.5, zy=0.5, fill_mode="reflect")
        return np.ascontiguousarray(img)


class ShiftHSV(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.AddToHueAndSaturation((20, 20), per_channel=False)
        self.solt_stream = slc.Stream([slt.ImageRandomHSV(p=1, h_range=(20, 20), s_range=(20, 20), v_range=(20, 20))])

    def albumentations(self, img):
        return albumentations.shift_hsv(img, hue_shift=20, sat_shift=20, val_shift=20)

    def torchvision_transform(self, img):
        img = torchvision.adjust_hue(img, hue_factor=0.1)
        img = torchvision.adjust_saturation(img, saturation_factor=1.2)
        img = torchvision.adjust_brightness(img, brightness_factor=1.2)
        return img


class Solarize(BenchmarkTest):
    def __init__(self):
        pass

    def albumentations(self, img):
        return albumentations.solarize(img)

    def pillow(self, img):
        return ImageOps.solarize(img)


class Equalize(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.AllChannelsHistogramEqualization()
        self.augmentor_op = Operations.HistogramEqualisation(probability=1)

    def albumentations(self, img):
        return albumentations.equalize(img)

    def pillow(self, img):
        return ImageOps.equalize(img)

    def imgaug(self, img):
        img = self.imgaug_transform.augment_image(img)
        return np.ascontiguousarray(img)


class RandomCrop64(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.CropToFixedSize(width=64, height=64)
        self.augmentor_op = Operations.Crop(probability=1, width=64, height=64, centre=False)
        self.solt_stream = slc.Stream([slt.CropTransform(crop_size=(64, 64), crop_mode="r")])

    def albumentations(self, img):
        img = albumentations.random_crop(img, crop_height=64, crop_width=64, h_start=0, w_start=0)
        return np.ascontiguousarray(img)

    def torchvision_transform(self, img):
        return torchvision.crop(img, i=0, j=0, h=64, w=64)

    def solt(self, img):
        dc = sld.DataContainer(img, "I")
        dc = self.solt_stream(dc)
        return np.ascontiguousarray(dc.data[0])


class RandomSizedCrop_64_512(BenchmarkTest):
    def __init__(self):
        self.augmentor_pipeline = Pipeline()
        self.augmentor_pipeline.add_operation(Operations.Crop(probability=1, width=64, height=64, centre=False))
        self.augmentor_pipeline.add_operation(
            Operations.Resize(probability=1, width=512, height=512, resample_filter="BILINEAR")
        )
        self.imgaug_transform = iaa.Sequential(
            [iaa.CropToFixedSize(width=64, height=64), iaa.Scale(size=512, interpolation="linear")]
        )
        self.solt_stream = slc.Stream(
            [slt.CropTransform(crop_size=(64, 64), crop_mode="r"), slt.ResizeTransform(resize_to=(512, 512))]
        )

    def albumentations(self, img):
        img = albumentations.random_crop(img, crop_height=64, crop_width=64, h_start=0, w_start=0)
        return albumentations.resize(img, height=512, width=512)

    def augmentor(self, img):
        for operation in self.augmentor_pipeline.operations:
            img, = operation.perform_operation([img])
        return np.array(img, np.uint8, copy=True)

    def torchvision_transform(self, img):
        img = torchvision.crop(img, i=0, j=0, h=64, w=64)
        return torchvision.resize(img, (512, 512))


class ShiftRGB(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Add((100, 100), per_channel=False)

    def albumentations(self, img):
        return albumentations.shift_rgb(img, r_shift=100, g_shift=100, b_shift=100)

    def keras(self, img):
        img = keras.apply_channel_shift(img, intensity=100, channel_axis=2)
        return np.ascontiguousarray(img)


class PadToSize512(BenchmarkTest):
    def __init__(self):
        self.solt_stream = slc.Stream([slt.PadTransform(pad_to=(512, 512), padding="r")])

    def albumentations(self, img):
        return albumentations.pad(img, min_height=512, min_width=512)

    def torchvision_transform(self, img):
        if img.size[0] < 512:
            img = torchvision.pad(img, (int((1 + 512 - img.size[0]) / 2), 0), padding_mode="reflect")
        if img.size[1] < 512:
            img = torchvision.pad(img, (0, int((1 + 512 - img.size[1]) / 2)), padding_mode="reflect")
        return img


class Resize512(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Scale(size=512, interpolation="linear")
        self.solt_stream = slc.Stream([slt.ResizeTransform(resize_to=(512, 512))])
        self.augmentor_op = Operations.Resize(probability=1, width=512, height=512, resample_filter="BILINEAR")

    def albumentations(self, img):
        return albumentations.resize(img, height=512, width=512)

    def torchvision_transform(self, img):
        return torchvision.resize(img, (512, 512))


class Gamma(BenchmarkTest):
    def __init__(self):
        self.solt_stream = slc.Stream([slt.ImageGammaCorrection(p=1, gamma_range=(0.5, 0.5))])

    def albumentations(self, img):
        return albumentations.gamma_transform(img, gamma=0.5)

    def torchvision_transform(self, img):
        return torchvision.adjust_gamma(img, gamma=0.5)


class Grayscale(BenchmarkTest):
    def __init__(self):
        self.augmentor_op = Operations.Greyscale(probability=1)
        self.imgaug_transform = iaa.Grayscale(alpha=1.0)
        self.solt_stream = slc.Stream([slt.ImageColorTransform(mode="rgb2gs")])

    def albumentations(self, img):
        return albumentations.to_gray(img)

    def torchvision_transform(self, img):
        return torchvision.to_grayscale(img, num_output_channels=3)

    def solt(self, img):
        dc = sld.DataContainer(img, "I")
        dc = self.solt_stream(dc)
        return cv2.cvtColor(dc.data[0], cv2.COLOR_GRAY2RGB)

    def augmentor(self, img):
        img = self.augmentor_op.perform_operation([img])[0]
        img = np.array(img, np.uint8, copy=True)
        return np.dstack([img, img, img])


class Posterize(BenchmarkTest):
    def albumentations(self, img):
        return albumentations.posterize(img, 4)

    def pillow(self, img):
        return ImageOps.posterize(img, 4)


class Multiply(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Multiply(mul=1.5)

    def albumentations(self, img):
        return albumentations.multiply(img, np.array([1.5]))


class MultiplyElementwise(BenchmarkTest):
    def __init__(self):
        self.aug = A.MultiplicativeNoise((0, 1), per_channel=True, elementwise=True, p=1)
        self.imgaug_transform = iaa.MultiplyElementwise(mul=(0, 1), per_channel=True)

    def albumentations(self, img):
        return self.aug(image=img)["image"]


def main():
    args = parse_args()
    package_versions = get_package_versions()
    if args.print_package_versions:
        print(package_versions)
    images_per_second = defaultdict(dict)
    libraries = args.libraries
    data_dir = args.data_dir
    paths = list(sorted(os.listdir(data_dir)))
    paths = paths[: args.images]
    imgs_cv2 = [read_img_cv2(os.path.join(data_dir, path)) for path in paths]
    imgs_pillow = [read_img_pillow(os.path.join(data_dir, path)) for path in paths]

    benchmarks = [
        HorizontalFlip(),
        VerticalFlip(),
        Rotate(),
        ShiftScaleRotate(),
        Brightness(),
        Contrast(),
        BrightnessContrast(),
        ShiftRGB(),
        ShiftHSV(),
        Gamma(),
        Grayscale(),
        RandomCrop64(),
        PadToSize512(),
        Resize512(),
        RandomSizedCrop_64_512(),
        Posterize(),
        Solarize(),
        Equalize(),
        Multiply(),
        MultiplyElementwise(),
    ]
    for library in libraries:
        imgs = imgs_pillow if library in ("torchvision", "augmentor", "pillow") else imgs_cv2
        pbar = tqdm(total=len(benchmarks))
        for benchmark in benchmarks:
            pbar.set_description("Current benchmark: {} | {}".format(library, benchmark))
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
    df = df.applymap(lambda r: format_results(r, args.show_std))
    df = df[libraries]
    augmentations = [str(i) for i in benchmarks]
    df = df.reindex(augmentations)
    if args.markdown:
        makedown_generator = MarkdownGenerator(df, package_versions)
        makedown_generator.print()
    else:
        print(df.head(len(augmentations)))


if __name__ == "__main__":
    main()
