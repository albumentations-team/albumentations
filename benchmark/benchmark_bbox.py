from __future__ import division, print_function

import argparse
from abc import ABC
from collections import defaultdict
from timeit import Timer
from typing import List

import cv2
import numpy as np
import pandas as pd
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from tqdm import tqdm

import albumentations as A

from benchmark.utils import set_bench_env_vars, get_package_versions, MarkdownGenerator, get_image, format_results

set_bench_env_vars()

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
        "-i", "--images", default=10, type=int, metavar="N", help="number of images for benchmarking (default: 200)"
    )
    parser.add_argument(
        "-b",
        "--bboxes",
        default=100,
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


def generate_random_bboxes(bbox_nums: int = 1):
    return np.sort(np.random.random(size=(bbox_nums, 4)))


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
                A.Perspective(p=0.5),
                A.Affine(scale=0.1, translate_percent=0.1, rotate=0.3, shear=0.2, p=0.5),
                A.PadIfNeeded(min_height=1024, min_width=1024, p=1.0),
            ],
            bbox_params=bbox_params,
        )
        self.imgaug_transform = iaa.Sequential(
            [
                iaa.HorizontalFlip(p=1),
                iaa.VerticalFlip(p=1),
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
    images_per_second = defaultdict(dict)
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
        PiecewiseAffine(),  # very slow
        Sequence(),
    ]
    imgs = [get_image(img_size=(100, 100, 3)) for _ in range(args.images)]
    bboxes = [generate_random_bboxes(args.bboxes) for _ in range(args.images)]

    for library in libraries:

        class_ids = [np.random.randint(low=0, high=1, size=args.bboxes) for _ in range(args.images)]
        pbar = tqdm(total=len(benchmarks))
        for benchmark in benchmarks:
            pbar.set_description("Current benchmark: {} | {}".format(library, benchmark))
            benchmark_images_per_second = None
            if benchmark.is_supported_by(library):
                timer = Timer(lambda: benchmark.run(library, imgs, bboxes=bboxes, class_ids=class_ids))
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
