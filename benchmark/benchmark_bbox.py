from __future__ import division, print_function

import argparse
import math
import os
import sys
from abc import ABC
from collections import defaultdict
from timeit import Timer

import cv2
import keras_preprocessing.image as keras
import numpy as np
import pandas as pd
import pkg_resources
import solt.core as slc
import solt.transforms as slt
import solt.utils as slu
import torchvision.transforms.functional as torchvision
from Augmentor import Operations, Pipeline
from imgaug import augmenters as iaa
from PIL import Image, ImageOps
from pytablewriter import MarkdownTableWriter
from pytablewriter.style import Style
from tqdm import tqdm

import albumentations as A
import albumentations.augmentations.functional as albumentations
from albumentations.augmentations.crops.functional import random_crop
from albumentations.augmentations.geometric.functional import (
    resize,
    rotate,
    shift_scale_rotate,
)

cv2.setNumThreads(0)  # noqa E402
cv2.ocl.setUseOpenCL(False)  # noqa E402

os.environ["OMP_NUM_THREADS"] = "1"  # noqa E402
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # noqa E402
os.environ["MKL_NUM_THREADS"] = "1"  # noqa E402
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # noqa E402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa E402


DEFAULT_BENCHMARKING_LIBRARIES = [
    "albumentations",
    "imgaug",
]


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

    def is_supported_by(self, library):
        if library == "imgaug":
            return hasattr(self, "imgaug_transform")
        return hasattr(self, library)

    def run(self, library, imgs):
        transform = getattr(self, library)
        for img in imgs:
            transform(img)


class HorizontalFlip(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Fliplr(p=1)
        self.augmentor_op = Operations.Flip(probability=1, top_bottom_left_right="LEFT_RIGHT")
        self.solt_stream = slc.Stream([slt.Flip(p=1, axis=1)])

    def albumentations(self, img):
        if img.ndim == 3 and img.shape[2] > 1 and img.dtype == np.uint8:
            return A.hflip_cv2(img)
        return A.hflip(img)

    def torchvision_transform(self, img):
        return torchvision.hflip(img)

    def keras(self, img):
        return np.ascontiguousarray(keras.flip_axis(img, axis=1))

    def imgaug(self, img):
        return np.ascontiguousarray(self.imgaug_transform.augment_image(img))
