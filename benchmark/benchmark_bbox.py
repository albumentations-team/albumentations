from __future__ import division, print_function

import argparse
import math
import os
import random
import sys
from abc import ABC
from collections import defaultdict
from timeit import Timer
from typing import List

import cv2
import numpy as np
import pandas as pd
import pkg_resources
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from pytablewriter import MarkdownTableWriter
from pytablewriter.style import Style
from tqdm import tqdm

import albumentations as A
from albumentations.augmentations.crops import functional as CFunc
from albumentations.augmentations.geometric import functional as GFunc

cv2.setNumThreads(0)  # noqa E402
cv2.ocl.setUseOpenCL(False)  # noqa E402

os.environ["OMP_NUM_THREADS"] = "1"  # noqa E402
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # noqa E402
os.environ["MKL_NUM_THREADS"] = "1"  # noqa E402
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # noqa E402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa E402


DEFAULT_BENCHMARKING_LIBRARIES = [
    "imgaug",
    "albumentations",
]

bbox_params = A.BboxParams(format="albumentations", label_fields=["class_id"], check_each_transform=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Augmentation libraries performance benchmark")
    parser.add_argument(
        "-l", "--libraries", default=DEFAULT_BENCHMARKING_LIBRARIES, nargs="+", help="list of libraries to benchmark"
    )
    parser.add_argument(
        "-i", "--images", default=100, type=int, metavar="N", help="number of images for benchmarking (default: 200)"
    )
    parser.add_argument(
        "-b",
        "--bboxes",
        default=20,
        type=int,
        help="number of bounding boxes in an image for benchmarking (default: 100)",
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
        "numpy",
        "opencv-python",
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
        libraries = ["Python", "numpy", "opencv-python"]
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


def read_img_cv2(img_size=(512, 512, 3)):
    img = np.zeros(shape=img_size, dtype=np.uint8)
    return img


def generate_random_bboxes(bbox_nums: int = 1):
    return np.sort(np.random.random(size=(bbox_nums, 4)))


def format_results(seconds_per_image_for_aug, show_std=False):
    if seconds_per_image_for_aug is None:
        return "-"
    result = str(np.round(np.mean(seconds_per_image_for_aug), 3))
    if show_std:
        result += " ± {}".format(math.ceil(np.std(seconds_per_image_for_aug)))
    return result


class BenchmarkTest(ABC):
    def __str__(self):
        return self.__class__.__name__

    def imgaug(self, img, bboxes, class_id):
        bbs = BoundingBoxesOnImage([BoundingBox(*bbox) for bbox in bboxes], shape=img.shape)
        img_aug, bbox_aug = self.imgaug_transform(image=img, bounding_boxes=bbs)
        np_bboxes = np.array([(bbox.x1, bbox.y1, bbox.x2, bbox.y2) for bbox in bbox_aug.bounding_boxes])
        return np.ascontiguousarray(img_aug), np_bboxes

    def is_supported_by(self, library):
        if library == "imgaug":
            return hasattr(self, "imgaug_transform")
        return hasattr(self, library)

    def run(self, library, imgs: List[np.ndarray], bboxes: List[np.ndarray], class_ids):
        transform = getattr(self, library)
        for img, bboxes_, class_id in zip(imgs, bboxes, class_ids):
            transform(img, bboxes_, class_id)


class HorizontalFlip(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Fliplr(p=1)
        self.alb_compose = A.Compose(
            [
                A.HorizontalFlip(p=1.0),
            ],
            bbox_params=bbox_params,
        )

    def albumentations(self, img, bboxes, class_id):
        return self.alb_compose(image=img, bboxes=bboxes, class_id=class_id)


class VerticalFlip(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Flipud(p=1)
        self.alb_compose = A.Compose([A.VerticalFlip(p=1.0)], bbox_params=bbox_params)

    def albumentations(self, img, bboxes, class_id):
        return self.alb_compose(image=img, bboxes=bboxes, class_id=class_id)


class Flip(BenchmarkTest):
    def __init__(self):
        self.alb_compose = A.Compose([A.Flip(p=1.0)], bbox_params=bbox_params)
        self.imgaug_transform = iaa.Sequential(
            [
                iaa.Fliplr(p=1),
                iaa.Flipud(p=1),
            ]
        )

    def albumentations(self, img, bboxes, class_id):
        return self.alb_compose(image=img, bboxes=bboxes, class_id=class_id)


class Rotate(BenchmarkTest):
    def __init__(self):

        self.imgaug_transform = iaa.Rotate()
        self.alb_compose = A.Compose([A.Rotate(p=1)], bbox_params=bbox_params)

    def albumentations(self, img, bboxes, class_id):
        return self.alb_compose(image=img, bboxes=bboxes, class_id=class_id)


class SafeRotate(BenchmarkTest):
    def __init__(self):
        self.alb_compose = A.Compose([A.SafeRotate(p=1.0)], bbox_params=bbox_params)

    def albumentations(self, img, bboxes, class_id):
        return self.alb_compose(image=img, bboxes=bboxes, class_id=class_id)


class ShiftScaleRotate(BenchmarkTest):
    def __init__(self):
        self.alb_compose = A.Compose(
            [
                A.ShiftScaleRotate(p=1.0),
            ],
            bbox_params=bbox_params,
        )

    def albumentations(self, img, bboxes, class_id):
        return self.alb_compose(image=img, bboxes=bboxes, class_id=class_id)


class Transpose(BenchmarkTest):
    def __init__(self):
        self.alb_compose = A.Compose(
            [
                A.Transpose(p=1),
            ],
            bbox_params=bbox_params,
        )

    def albumentations(self, img, bboxes, class_id):
        return self.alb_compose(image=img, bboxes=bboxes, class_id=class_id)


class Pad(BenchmarkTest):
    def __init__(self):
        self.alb_compose = A.Compose([A.PadIfNeeded(min_height=1024, min_width=1024, p=1.0)], bbox_params=bbox_params)
        self.imgaug_transform = iaa.CenterPadToFixedSize(width=1024, height=1024)

    def albumentations(self, img, bboxes, class_id):
        return self.alb_compose(image=img, bboxes=bboxes, class_id=class_id)


class RandomRotate90(BenchmarkTest):
    def __init__(self):
        self.alb_compose = A.Compose([A.RandomRotate90(p=1.0)], bbox_params=bbox_params)
        self.imgaug_transform = iaa.Rot90()

    def albumentations(self, img, bboxes, class_id):
        return self.alb_compose(image=img, bboxes=bboxes, class_id=class_id)


class Perspective(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.PerspectiveTransform(scale=(0.05, 1))
        self.alb_compose = A.Compose([A.Perspective(p=1.0)], bbox_params=bbox_params)
        self.imgaug_transform = iaa.PerspectiveTransform(
            scale=(0.05, 0.1),
        )

    def albumentations(self, img, bboxes, class_id):
        return self.alb_compose(image=img, bboxes=bboxes, class_id=class_id)


class RandomCropNearBBox(BenchmarkTest):
    def __init__(self):
        self.alb_compose = A.Compose(
            [
                A.RandomCropNearBBox(p=1.0),
            ],
            bbox_params=bbox_params,
        )

    def albumentations(self, img, bboxes, class_id):
        return self.alb_compose(image=img, bboxes=bboxes, class_id=class_id, cropping_bbox=[0, 5, 10, 20])


class BBoxSafeRandomCrop(BenchmarkTest):
    def __init__(self):
        self.alb_compose = A.Compose([A.BBoxSafeRandomCrop(p=1.0)], bbox_params=bbox_params)

    def albumentations(self, img, bboxes, class_id):
        return self.alb_compose(image=img, bboxes=bboxes, class_id=class_id)


class CenterCrop(BenchmarkTest):
    def __init__(self):
        self.alb_compose = A.Compose([A.CenterCrop(10, 10, p=1.0)], bbox_params=bbox_params)

    def albumentations(self, img, bboxes, class_id):
        return self.alb_compose(image=img, bboxes=bboxes, class_id=class_id)


class Crop(BenchmarkTest):
    def __init__(self):
        self.alb_compose = A.Compose([A.Crop(x_max=100, y_max=100, p=1.0)], bbox_params=bbox_params)

        self.imgaug_transform = iaa.Crop()

    def albumentations(self, img, bboxes, class_id):
        return self.alb_compose(image=img, bboxes=bboxes, class_id=class_id)


class CropAndPad(BenchmarkTest):
    def __init__(self):
        self.alb_compose = A.Compose([A.CropAndPad(percent=0.1, p=1.0)], bbox_params=bbox_params)

    def albumentations(self, img, bboxes, class_id):
        return self.alb_compose(image=img, bboxes=bboxes, class_id=class_id)


class RandomCropFromBorders(BenchmarkTest):
    def __init__(self):
        self.alb_compose = A.Compose([A.RandomCropFromBorders(p=1.0)], bbox_params=bbox_params)

    def albumentations(self, img, bboxes, class_id):
        return self.alb_compose(image=img, bboxes=bboxes, class_id=class_id)


class Affine(BenchmarkTest):
    def __init__(self):
        self.alb_compose = A.Compose(
            [A.Affine(scale=0.1, translate_percent=0.1, rotate=0.3, shear=0.2, p=1.0)], bbox_params=bbox_params
        )

        self.imgaug_transform = iaa.Affine(scale=0.1, translate_percent=0.1, rotate=0.3, shear=0.2)

    def albumentations(self, img, bboxes, class_id):
        return self.alb_compose(image=img, bboxes=bboxes, class_id=class_id)


class PiecewiseAffine(BenchmarkTest):
    def __init__(self):

        scale = (0.03, 0.05)
        nb_rows = 4
        nb_cols = 4

        self.alb_compose = A.Compose(
            [
                A.PiecewiseAffine(scale=scale, nb_rows=nb_rows, nb_cols=nb_cols),
            ],
            bbox_params=bbox_params,
        )

        self.imgaug_transform = iaa.PiecewiseAffine(scale=scale, nb_rows=nb_rows, nb_cols=nb_cols)

    def albumentations(self, img, bboxes, class_id):
        return self.alb_compose(image=img, bboxes=bboxes, class_id=class_id)


class Sequence(BenchmarkTest):
    def __init__(self):
        self.alb_compose = A.Compose(
            [
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.Rotate(p=1, border_mode=cv2.BORDER_CONSTANT),
                A.RandomRotate90(p=1),
                A.Perspective(p=1),
                A.Affine(scale=0.1, translate_percent=0.1, rotate=0.3, shear=0.2, p=1.0),
                A.PadIfNeeded(min_height=1024, min_width=1024, p=1.0),
            ],
            bbox_params=bbox_params,
        )
        self.imgaug_transform = iaa.Sequential(
            [
                iaa.HorizontalFlip(),
                iaa.VerticalFlip(),
                iaa.Rotate(),
                iaa.Rot90(),
                iaa.PerspectiveTransform(
                    scale=(0.05, 0.1),
                ),
                iaa.Affine(scale=0.1, translate_percent=0.1, rotate=0.3, shear=0.2),
                iaa.CenterPadToFixedSize(width=1024, height=1024),
            ]
        )

    def albumentations(self, img, bboxes, class_id):
        return self.alb_compose(image=img, bboxes=bboxes, class_id=class_id)


def main():
    args = parse_args()
    package_versions = get_package_versions()
    if args.print_package_versions:
        print(package_versions)
    seconds_per_image = defaultdict(dict)
    libraries = args.libraries

    benchmarks = [
        HorizontalFlip(),
        VerticalFlip(),
        Flip(),
        Rotate(),
        SafeRotate(),
        RandomRotate90(),
        ShiftScaleRotate(),
        Transpose(),
        Pad(),
        Perspective(),
        RandomCropNearBBox(),
        BBoxSafeRandomCrop(),
        CenterCrop(),
        Crop(),
        CropAndPad(),
        RandomCropFromBorders(),
        Affine(),
        PiecewiseAffine(),
        Sequence(),
    ]
    imgs = [read_img_cv2(img_size=(512, 512, 3)) for _ in range(args.images)]
    bboxes = [generate_random_bboxes(args.bboxes) for _ in range(args.images)]

    for library in libraries:

        class_ids = [np.random.randint(low=0, high=1, size=args.bboxes) for _ in range(args.images)]
        pbar = tqdm(total=len(benchmarks))
        for benchmark in benchmarks:
            pbar.set_description("Current benchmark: {} | {}".format(library, benchmark))
            benchmark_second_per_image = None
            if benchmark.is_supported_by(library):
                timer = Timer(lambda: benchmark.run(library, imgs, bboxes=bboxes, class_ids=class_ids))
                run_times = timer.repeat(number=1, repeat=args.runs)
                benchmark_second_per_image = [run_time * 1000 / args.images for run_time in run_times]
            seconds_per_image[library][str(benchmark)] = benchmark_second_per_image
            pbar.update(1)
        pbar.close()
    pd.set_option("display.width", 1000)
    df = pd.DataFrame.from_dict(seconds_per_image)
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
