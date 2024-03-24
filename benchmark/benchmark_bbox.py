from __future__ import division, print_function

import argparse
from abc import ABC
from collections import defaultdict
from timeit import Timer
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from tqdm import tqdm
import torch
from torchvision.tv_tensors import BoundingBoxes as TorchBoundingBoxes
from torchvision.transforms import v2, InterpolationMode

import albumentations as A
from albumentations.core.bbox_utils import convert_bboxes_from_albumentations

from benchmark.utils import (
    set_bench_env_vars,
    get_package_versions,
    MarkdownGenerator,
    get_image,
    format_results,
    get_markdown_table,
)

set_bench_env_vars()

DEFAULT_BENCHMARKING_LIBRARIES = [
    "imgaug",
    "albumentations",
    "torchvision",
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
        return self.imgaug_transform(image=img, bounding_boxes=bboxes)

    def torchvision(self, img: torch.Tensor, bboxes, class_id) -> torch.Tensor:
        return self.torchvision_transform(img, bboxes)[0].contiguous()

    def is_supported_by(self, library):
        if library == "imgaug":
            return hasattr(self, "imgaug_transform")
        elif library == "torchvision":
            return hasattr(self, "torchvision_transform")
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

    def torchvision_transform(self, img: torch.Tensor, bboxes: TorchBoundingBoxes):
        return v2.RandomHorizontalFlip(p=1)(img, bboxes)


class VerticalFlip(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Flipud(p=1)
        self.alb_compose = A.Compose([A.VerticalFlip(p=1.0)], bbox_params=bbox_params)

    def albumentations(self, img, bboxes, class_id):
        return self.alb_compose(image=img, bboxes=bboxes, class_id=class_id)

    def torchvision_transform(self, img: torch.Tensor, bboxes: TorchBoundingBoxes):
        return v2.RandomVerticalFlip(p=1)(img, bboxes)


class Rotate(BenchmarkTest):
    def __init__(self):
        self.angle = 45
        self.imgaug_transform = iaa.Rotate(rotate=(self.angle, self.angle), order=1, mode="reflect")
        self.alb_compose = A.Compose([A.Rotate(limit=self.angle, p=1)], bbox_params=bbox_params)

    def albumentations(self, img, bboxes, class_id):
        return self.alb_compose(image=img, bboxes=bboxes, class_id=class_id)

    def torchvision_transform(self, img: torch.Tensor, bboxes: TorchBoundingBoxes):
        return v2.RandomRotation(degrees=self.angle, interpolation=InterpolationMode.BILINEAR)(img, bboxes)


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
        self.scale = (0.05, 0.1)
        self.alb_compose = A.Compose([A.Perspective(scale=self.scale, p=1.0)], bbox_params=bbox_params)
        self.imgaug_transform = iaa.PerspectiveTransform(
            scale=self.scale,
        )

    def albumentations(self, img, bboxes, class_id):
        return self.alb_compose(image=img, bboxes=bboxes, class_id=class_id)

    def torchvision_transform(self, img: torch.Tensor, bboxes: TorchBoundingBoxes):
        return v2.RandomPerspective(
            distortion_scale=self.scale[1], interpolation=InterpolationMode.BILINEAR, p=1
        )(img, bboxes)


class Crop(BenchmarkTest):
    def __init__(self):
        self.alb_compose = A.Compose([A.Crop(x_max=100, y_max=100, p=1.0)], bbox_params=bbox_params)
        self.imgaug_transform = iaa.Crop()

    def albumentations(self, img, bboxes, class_id):
        return self.alb_compose(image=img, bboxes=bboxes, class_id=class_id)


class Affine(BenchmarkTest):
    def __init__(self):
        self.angle = 25.0
        self.shift = (50, 50)
        self.scale = 2.0
        self.shear = [10.0, 15.0]

        self.imgaug_transform = iaa.Affine(
            scale=(self.scale, self.scale),
            rotate=(self.angle, self.angle),
            translate_px=self.shift,
            order=1,
            mode="reflect",
        )

    def albumentations(self, img, bboxes, class_id):
        height, width = img.shape[-2], img.shape[-1]
        alb_compose = A.Compose(
            [
                A.Affine(
                    translate_percent=[self.shift[0] / width, self.shift[1] / height],
                    rotate=self.angle,
                    shear=self.shear,
                    interpolation=cv2.INTER_LINEAR,
                )
            ], bbox_params=bbox_params
        )
        return alb_compose(image=img, bboxes=bboxes, class_id=class_id)

    def torchvision_transform(self, img: torch.Tensor, bboxes: TorchBoundingBoxes):
        height, width = img.shape[-2], img.shape[-1]
        return v2.RandomAffine(
            degrees=self.angle,
            translate=[self.shift[0] / width, self.shift[1] / height],
            scale=(self.scale, self.scale),
            shear=self.shear,
            interpolation=InterpolationMode.BILINEAR,
        )(img, bboxes)


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
        print(get_markdown_table(package_versions))

    images_per_second = defaultdict(dict)
    libraries = args.libraries

    benchmarks = [
        HorizontalFlip(),
        VerticalFlip(),
        Rotate(),
        RandomRotate90(),
        Pad(),
        Perspective(),
        Crop(),
        Affine(),
        PiecewiseAffine(),  # very slow
        Sequence(),
    ]

    def get_imgaug_bboxes(img: np.ndarray, bboxes: np.ndarray) -> BoundingBoxesOnImage:
        return BoundingBoxesOnImage([BoundingBox(*bbox) for bbox in bboxes], shape=img.shape)

    def get_torch_bboxes(img: np.ndarray, bboxes: np.ndarray) -> TorchBoundingBoxes:
        bboxes = convert_bboxes_from_albumentations(
            bboxes, "pascal_voc", img.shape[0], img.shape[1], check_validity=True
        )
        bboxes = torch.tensor(bboxes)
        return TorchBoundingBoxes(bboxes, format="XYXY", canvas_size=(img.shape[0], img.shape[1]))

    imgs_cv2 = [get_image(img_size=(100, 100, 3)) for _ in range(args.images)]
    imgs_torch = [torch.from_numpy(i.transpose(2, 0, 1)) for i in imgs_cv2]
    bboxes_albu = [generate_random_bboxes(args.bboxes) for _ in range(args.images)]
    bboxes_imgaug = [get_imgaug_bboxes(img, bboxes) for img, bboxes in zip(imgs_cv2, bboxes_albu)]
    bboxes_torch = [get_torch_bboxes(img, bboxes) for img, bboxes in zip(imgs_cv2, bboxes_albu)]

    def get_imgs_bboxes(library: str) -> Tuple[list, list]:
        if library == "torchvision":
            return imgs_torch, bboxes_torch
        if library == "imgaug":
            return imgs_cv2, bboxes_imgaug
        return imgs_cv2, bboxes_albu

    for library in libraries:
        class_ids = [np.random.randint(low=0, high=1, size=args.bboxes) for _ in range(args.images)]
        pbar = tqdm(total=len(benchmarks))
        for benchmark in benchmarks:
            pbar.set_description("Current benchmark: {} | {}".format(library, benchmark))
            benchmark_images_per_second = None

            imgs, bboxes = get_imgs_bboxes(library)

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
