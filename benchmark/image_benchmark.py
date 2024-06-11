import argparse
import copy
import os
import random
import sys
from collections import defaultdict
from contextlib import suppress
from pathlib import Path
from timeit import Timer
from typing import Any, Dict, List, Union

import augly.image as imaugs
import cv2
import kornia as K
import kornia.augmentation as Kaug
import numpy as np
import pandas as pd
import pkg_resources
import torch
from imgaug import augmenters as iaa
from PIL import Image
from torchvision.transforms import InterpolationMode, v2
from tqdm import tqdm

import albumentations as A
from benchmark.utils import (
    MarkdownGenerator,
    format_results,
    get_markdown_table,
    read_img_cv2,
    read_img_kornia,
    read_img_pillow,
    read_img_torch,
)

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

torch.set_num_threads(1)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


DEFAULT_BENCHMARKING_LIBRARIES = ["albumentations", "torchvision", "kornia", "augly", "imgaug"]


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
        "numpy",
        "opencv-python-headless",
        "scikit-image",
        "scipy",
        "pillow",
        "kornia",
        "augly",
    ]
    package_versions = {"Python": sys.version}
    for package in packages:
        with suppress(pkg_resources.DistributionNotFound):
            package_versions[package] = pkg_resources.get_distribution(package).version
    return package_versions


class BenchmarkTest:
    def __str__(self) -> str:
        return self.__class__.__name__

    def albumentations(self, img: np.ndarray) -> np.ndarray:
        img = self.albumentations_transform(img)
        return np.array(img, np.uint8, copy=False)

    def imgaug(self, img: np.ndarray) -> np.ndarray:
        img = self.imgaug_transform.augment_image(img)
        return np.array(img, np.uint8, copy=False)

    def torchvision(self, img: torch.Tensor) -> torch.Tensor:
        return self.torchvision_transform(img).contiguous()

    def kornia(self, img: torch.Tensor) -> torch.Tensor:
        return self.kornia_transform(img).contiguous()

    def augly(self, img: Image.Image) -> Image.Image:
        return self.augly_transform(img)

    def is_supported_by(self, library: str) -> bool:
        library_attr_map = {
            "imgaug": "imgaug_transform",
            "kornia": "kornia_transform",
            "torchvision": "torchvision_transform",
            "albumentations": "albumentations_transform",
            "augly": "augly_transform",
        }

        # Check if the library is in the map
        if library in library_attr_map:
            attrs = library_attr_map[library]
            # Ensure attrs is a list for uniform processing
            if not isinstance(attrs, list):
                attrs = [attrs]  # type: ignore[list-item]
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

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        return A.HorizontalFlip(p=1)(image=img)["image"]

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return v2.RandomHorizontalFlip(p=1)(img)

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomHorizontalFlip(p=1)(img)

    def augly_transform(self, img: Image.Image) -> Image.Image:
        return imaugs.HFlip(p=1)(img)


class VerticalFlip(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Flipud(p=1)

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        return A.VerticalFlip(p=1)(image=img)["image"]

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return v2.RandomVerticalFlip(p=1)(img)

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomVerticalFlip(p=1)(img)

    def augly_transform(self, img: Image.Image) -> Image.Image:
        return imaugs.VFlip(p=1)(img)


class Rotate(BenchmarkTest):
    def __init__(self):
        self.angle = 45
        self.imgaug_transform = iaa.Affine(rotate=(self.angle, self.angle), order=1, mode="reflect")

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        return A.Rotate(limit=self.angle, p=1)(image=img)["image"]

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return v2.RandomRotation(degrees=self.angle, interpolation=InterpolationMode.BILINEAR)(img)

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        # Convert degrees to radians for rotation
        angle = torch.tensor(-self.angle) * (torch.pi / 180.0)

        # Perform rotation
        return K.geometry.transform.rotate(img, angle=angle, mode="bilinear", padding_mode="zeros")

    def augly_transform(self, img: Image.Image) -> Image.Image:
        return imaugs.RandomRotation(min_degrees=self.angle, max_degrees=self.angle, p=1)(img)


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

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        height, width = img.shape[-2], img.shape[-1]
        return A.Affine(
            translate_percent=[self.shift[0] / width, self.shift[1] / height],
            rotate=self.angle,
            shear=self.shear,
            interpolation=cv2.INTER_LINEAR,
        )(image=img)["image"]

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        height, width = img.shape[-2], img.shape[-1]
        return v2.RandomAffine(
            degrees=self.angle,
            translate=[self.shift[0] / width, self.shift[1] / height],
            scale=(self.scale, self.scale),
            shear=self.shear,
            interpolation=InterpolationMode.BILINEAR,
        )(img)

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        batch_size = img.shape[0]
        scale = torch.tensor([[self.scale, self.scale]]).repeat(batch_size, 1)
        angle = torch.tensor([self.angle]).float()
        translation = torch.tensor([self.shift]).float()
        shear = torch.tensor([self.shear]).float()

        return K.geometry.transform.Affine(angle=angle, scale_factor=scale, translation=translation, shear=shear)(img)


class Equalize(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.AllChannelsHistogramEqualization()

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        return A.Equalize(p=1)(image=img)["image"]

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return v2.RandomEqualize(p=1)(img)

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomEqualize(p=1)(img)


class RandomCrop64(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.CropToFixedSize(width=64, height=64)

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        return A.RandomCrop(height=64, width=64, p=1)(image=img)["image"]

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return v2.RandomCrop(size=(64, 64))(img)

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomCrop(size=(64, 64), p=1)(img)

    def augly_transform(self, img: Image.Image) -> Image.Image:
        x1 = 0.25
        x2 = random.uniform(0.25, 1)
        y1 = 0.25
        y2 = random.uniform(0.25, 1)
        return imaugs.Crop(x1=x1, y1=y1, x2=x2, y2=y2, p=1)(img)


class RandomResizedCrop(BenchmarkTest):
    def __init__(
        self,
        height: int = 512,
        width: int = 512,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (0.75, 1.3333333333333333),
    ):
        self.height = height
        self.width = width
        self.ratio = ratio
        self.scale = scale

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        return A.RandomResizedCrop(
            scale=self.scale, height=self.height, width=self.width, ratio=self.ratio, interpolation=cv2.INTER_LINEAR
        )(image=img)["image"]

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return v2.RandomResizedCrop(
            scale=self.scale, size=[self.height, self.width], ratio=self.ratio, interpolation=InterpolationMode.BILINEAR
        )(img)

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomResizedCrop(size=[self.height, self.width], scale=self.scale, ratio=self.ratio)(img)


class ShiftRGB(BenchmarkTest):
    def __init__(self, pixel_shift: int = 100):
        self.pixel_shift = pixel_shift
        self.imgaug_transform = iaa.Add((pixel_shift, pixel_shift), per_channel=True)

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        return A.RGBShift(
            r_shift_limit=self.pixel_shift, g_shift_limit=self.pixel_shift, b_shift_limit=self.pixel_shift, p=1
        )(image=img)["image"]

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomRGBShift(self.pixel_shift / 255, self.pixel_shift / 255, self.pixel_shift / 255, p=1)(img)


class Resize(BenchmarkTest):
    def __init__(self, target_size: int = 512):
        self.target_size = target_size
        self.imgaug_transform = iaa.Resize(size=target_size, interpolation="linear")

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        return A.Resize(height=self.target_size, width=self.target_size, interpolation=cv2.INTER_LINEAR)(image=img)[
            "image"
        ]

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return v2.Resize(size=(self.target_size, self.target_size), interpolation=InterpolationMode.BILINEAR)(img)

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.Resize(size=(self.target_size, self.target_size))(img)

    def augly_transform(self, img: Image.Image) -> Image.Image:
        return imaugs.Resize(width=self.target_size, height=self.target_size, resample=Image.BILINEAR, p=1)(img)


class RandomGamma(BenchmarkTest):
    def __init__(self, gamma: float = 120):
        self.gamma = gamma
        self.imgaug_transform = iaa.GammaContrast(gamma=gamma / 100)

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        return A.RandomGamma(gamma_limit=(self.gamma, self.gamma), p=1)(image=img)["image"]

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return v2.functional.adjust_gamma(img, gamma=self.gamma / 100)

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomGamma(p=1.0, gamma=(self.gamma / 100, self.gamma / 100))(img)


class Grayscale(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Grayscale(alpha=1.0)

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        return A.to_gray(img)

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomGrayscale(p=1.0)(img)

    def augly_transform(self, img: Image.Image) -> Image.Image:
        return imaugs.Grayscale(p=1)(img)

    def torchvision_transform(self, img: torch.Tensor) -> np.ndarray:
        return v2.Grayscale(num_output_channels=3)(img)


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

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return v2.ColorJitter(
            brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue
        )(img)

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
        self.scale = (0.05, 0.1)
        self.imgaug_transform = iaa.PerspectiveTransform(scale=self.scale)

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        transform = A.Perspective(scale=self.scale, p=1, interpolation=cv2.INTER_LINEAR)  # Adjust scale as needed
        return transform(image=img)["image"]

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomPerspective(distortion_scale=self.scale[1], p=1)(img)

    def torchvision_transform(self, img: torch.Tensor) -> np.ndarray:
        return v2.RandomPerspective(distortion_scale=self.scale[1], interpolation=InterpolationMode.BILINEAR, p=1)(img)


class GaussianBlur(BenchmarkTest):
    def __init__(self, sigma: float = 2.0):
        self.sigma = sigma
        self.imgaug_transform = iaa.GaussianBlur(sigma=self.sigma)

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        transform = A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(self.sigma, self.sigma), p=1)
        return transform(image=img)["image"]

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return v2.GaussianBlur(kernel_size=[5, 5], sigma=(self.sigma, self.sigma))(img)

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomGaussianBlur(kernel_size=(5, 5), sigma=(self.sigma, self.sigma), p=1)(img)

    def augly_transform(self, img: Image.Image) -> Image.Image:
        return imaugs.Blur(radius=self.sigma, p=1)(img)


class MedianBlur(BenchmarkTest):
    def __init__(self, blur_limit: int = 5):
        # blur_limit or kernel size for median blur, ensuring it's an odd number
        self.blur_limit = blur_limit if blur_limit % 2 != 0 else blur_limit + 1
        self.imgaug_transform = iaa.MedianBlur(k=(self.blur_limit, self.blur_limit))

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
        transform = A.MotionBlur(blur_limit=self.kernel_size, p=1)
        return transform(image=img)["image"]

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomMotionBlur(kernel_size=self.kernel_size, angle=self.angle, direction=self.direction, p=1)(img)


class Posterize(BenchmarkTest):
    def __init__(self, bits: int = 4):
        self.bits = bits  # Number of bits to keep for each color channel
        self.imgaug_transform = iaa.Posterize(nb_bits=self.bits)

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        transform = A.Posterize(num_bits=self.bits, p=1)
        return transform(image=img)["image"]

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomPosterize(bits=self.bits, p=1)(img)

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return v2.RandomPosterize(bits=self.bits, p=1)(img)


class JpegCompression(BenchmarkTest):
    def __init__(self, quality: int = 50):
        self.quality = quality
        self.imgaug_transform = iaa.JpegCompression(compression=self.quality)

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        transform = A.ImageCompression(quality_lower=self.quality, quality_upper=self.quality, p=1)
        return transform(image=img)["image"]

    def augly_transform(self, img: Image.Image) -> Image.Image:
        return imaugs.EncodingQuality(quality=self.quality, p=1)(img)


class GaussianNoise(BenchmarkTest):
    def __init__(self, mean: float = 127, var: float = 0.010):
        self.mean = mean
        self.var = var
        self.imgaug_transform = iaa.AdditiveGaussianNoise(loc=self.mean, scale=(0, self.var * 255))

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        transform = A.GaussNoise(var_limit=self.var * 255, mean=self.mean, p=1)
        return transform(image=img)["image"]

    def augly_transform(self, img: Image.Image) -> Image.Image:
        return imaugs.RandomNoise(mean=self.mean, var=self.var * 255)(img)


class Elastic(BenchmarkTest):
    def __init__(self, alpha: float = 50.0, sigma: float = 5.0):
        # Parameters: alpha controls the intensity of the deformation, sigma controls the smoothness
        self.alpha = alpha
        self.sigma = sigma
        self.imgaug_transform = iaa.ElasticTransformation(alpha=self.alpha, sigma=self.sigma)

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        transform = A.ElasticTransform(alpha=self.alpha, sigma=self.sigma, p=1, approximate=True)
        return transform(image=img)["image"]

    def kornia_transform(self, img: torch.Tensor) -> np.ndarray:
        sigma = torch.tensor((self.sigma, self.sigma)).float()
        alpha = torch.tensor([self.alpha, self.alpha]).float()
        return Kaug.RandomElasticTransform(alpha=alpha, sigma=sigma, p=1)(img)

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        # Uses approximate Elastic by default
        return v2.ElasticTransform(alpha=self.alpha, sigma=self.sigma, interpolation=InterpolationMode.BILINEAR)(img)


class Normalize(BenchmarkTest):
    def __init__(self):
        self.mean = (0.485, 0.456, 0.406)
        self.std=(0.229, 0.224, 0.225)

    def albumentations_transform(self, img: torch.Tensor) -> np.ndarray:
        transform = A.Normalize(mean=self.mean, std=self.std, p=1)
        return transform(image=img)["image"]

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.Normalize(mean=self.mean, std=self.std, p=1)(img)

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return v2.Normalize(mean=self.mean, std=self.std)(img.float())


def run_benchmarks(benchmarks: List[BenchmarkTest], args: argparse.Namespace, libraries: List[str], data_dir: Path) -> Dict[str, Dict[str, Any]]:
    images_per_second: Dict[str, Dict[str, Any]] = defaultdict(dict)

    paths = sorted(data_dir.glob("*.*"))
    paths = paths[: args.images]
    imgs_cv2 = [read_img_cv2(path) for path in tqdm(paths, desc="Loading images for OpenCV")]
    imgs_pillow = [read_img_pillow(path) for path in tqdm(paths, desc="Loading images for Pillow")]
    imgs_torch = [read_img_torch(path) for path in tqdm(paths, desc="Loading images for Torch")]
    imgs_kornia = [read_img_kornia(path) for path in tqdm(paths, desc="Loading images for Kornia")]

    def get_imgs(library: str) -> list:
        if library == "augly":
            return imgs_pillow
        if library == "torchvision":
            return imgs_torch
        if library == "kornia":
            return imgs_kornia
        return imgs_cv2

    pbar = tqdm(total=len(benchmarks))

    for benchmark in benchmarks:
        shuffled_libraries = copy.deepcopy(libraries)  # Create a deep copy of the libraries list
        random.shuffle(shuffled_libraries)  # Shuffle the copied list
        pbar.set_description(f"Current benchmark: {benchmark}")

        for library in shuffled_libraries:
            imgs = get_imgs(library)

            benchmark_images_per_second = None

            if benchmark.is_supported_by(library):
                timer = Timer(lambda: benchmark.run(library, imgs))
                run_times = timer.repeat(number=1, repeat=args.runs)
                benchmark_images_per_second = [1 / (run_time / args.images) for run_time in run_times]
            images_per_second[library][str(benchmark)] = benchmark_images_per_second

        pbar.update(1)
    pbar.close()

    return images_per_second


def main() -> None:
    benchmarks = [
        HorizontalFlip(),
        VerticalFlip(),
        Rotate(),
        Affine(),
        Equalize(),
        RandomCrop64(),
        RandomResizedCrop(),
        ShiftRGB(),
        Resize(512),
        RandomGamma(),
        Grayscale(),
        ColorJitter(),
        RandomPerspective(),
        GaussianBlur(),
        MedianBlur(),
        MotionBlur(),
        Posterize(),
        JpegCompression(),
        GaussianNoise(),
        Elastic(),
        Normalize()
    ]

    args = parse_args()
    package_versions = get_package_versions()
    if args.print_package_versions:
        print(get_markdown_table(package_versions))

    libraries = args.libraries
    data_dir = Path(args.data_dir)

    images_per_second = run_benchmarks(benchmarks, args, libraries, data_dir)

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
        print(df)

if __name__ == "__main__":
    main()
