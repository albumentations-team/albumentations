import argparse
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
from Augmentor import Operations, Pipeline
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

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


DEFAULT_BENCHMARKING_LIBRARIES = [
    "albumentations",
    "torchvision",
    "kornia",
    "augly",
    "imgaug",
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
    def __str__(self) -> str:
        return self.__class__.__name__

    def albumentations(self, img: np.ndarray) -> np.ndarray:
        img = self.albumentations_transform(img)
        return np.array(img, np.uint8, copy=False)

    def imgaug(self, img: np.ndarray) -> np.ndarray:
        img = self.imgaug_transform.augment_image(img)
        return np.array(img, np.uint8, copy=False)

    def augmentor(self, img: Image.Image) -> Image.Image:
        return self.augmentor_op.perform_operation([img])[0]

    def torchvision(self, img: torch.Tensor) -> torch.Tensor:
        return self.torchvision_transform(img).contiguous()

    def kornia(self, img: torch.Tensor) -> torch.Tensor:
        return self.kornia_transform(img).contiguous()

    def augly(self, img: Image.Image) -> Image.Image:
        return self.augly_transform(img)

    def is_supported_by(self, library: str) -> bool:
        library_attr_map = {
            "imgaug": "imgaug_transform",
            "augmentor": ["augmentor_op", "augmentor_pipeline"],
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
        self.augmentor_op = Operations.Flip(probability=1, top_bottom_left_right="LEFT_RIGHT")

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
        self.augmentor_op = Operations.Flip(probability=1, top_bottom_left_right="TOP_BOTTOM")

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
        self.augmentor_op = Operations.RotateStandard(probability=1, max_left_rotation=45, max_right_rotation=45)

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
        height, width = img.shape[-2], img.shape[-1]
        return v2.RandomAffine(
            degrees=self.angle,
            translate=[self.shift[0] / width, self.shift[1] / height],
            scale=(self.scale, self.scale),
            shear=0,
            interpolation=InterpolationMode.BILINEAR,
        )(img)

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
        return v2.RandomEqualize(p=1)(img)

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomEqualize(p=1)(img)


class RandomCrop64(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.CropToFixedSize(width=64, height=64)
        self.augmentor_op = Operations.Crop(probability=1, width=64, height=64, centre=False)

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
        transform = v2.Compose(
            [v2.RandomCrop(size=(64, 64)), v2.Resize(size=(512, 512), interpolation=InterpolationMode.BILINEAR)]
        )
        return transform(img)

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomResizedCrop(size=(512, 512), scale=(0.08, 1.0), ratio=(0.75, 1.33))(img)


class ShiftRGB(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Add((100, 100), per_channel=False)

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        return A.RGBShift(r_shift_limit=100, g_shift_limit=100, b_shift_limit=100, p=1)(image=img)["image"]

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomRGBShift(0.5, 0.5, 0.5, p=1)(img)  # Define the shifts for R, G, B channels


class Resize512(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.Resize(size=512, interpolation="linear")
        self.augmentor_op = Operations.Resize(probability=1, width=512, height=512, resample_filter="BILINEAR")

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        return A.Resize(height=512, width=512, interpolation=cv2.INTER_LINEAR)(image=img)["image"]

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return v2.Resize(size=(512, 512), interpolation=InterpolationMode.BILINEAR)(img)

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.Resize(size=(512, 512))(img)

    def augly_transform(self, img: Image.Image) -> Image.Image:
        return imaugs.Resize(width=512, height=512, resample=Image.BILINEAR, p=1)(img)


class RandomGamma(BenchmarkTest):
    def __init__(self):
        self.imgaug_transform = iaa.GammaContrast(gamma=0.5)

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        return A.RandomGamma(gamma_limit=(80, 120), p=1)(image=img)["image"]

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return v2.functional.adjust_gamma(img, gamma=0.5)

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

    def torchvision(self, img: torch.Tensor) -> torch.Tensor:
        return v2.ColorJitter(
            brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue, p=1
        )

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

    def kornia_transform(self, img: torch.Tensor) -> torch.Tensor:
        return Kaug.RandomPosterize(bits=self.bits, p=1)(img)

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return v2.RandomPosterize(bits=self.bits, p=1)(img)


class JpegCompression(BenchmarkTest):
    def __init__(self, quality: int = 50):
        # Quality: Value between 0 and 100 (higher means better). In imgaug, it's 0-100 scale.
        self.quality = quality
        self.imgaug_transform = iaa.JpegCompression(compression=self.quality)

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        transform = A.ImageCompression(quality_lower=self.quality, quality_upper=self.quality, always_apply=True)
        return transform(image=img)["image"]

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
        return imaugs.RandomNoise(mean=self.mean, var=self.var)(img)

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        # Ensure img tensor is in float format; assume it is scaled between 0 and 1
        noise = torch.randn(img.size()) * self.var**0.5 + self.mean
        img_noisy = img + noise
        return torch.clamp(img_noisy, 0.0, 1.0)


class Elastic(BenchmarkTest):
    def __init__(self, alpha: float = 50.0, sigma: float = 5.0):
        # Parameters: alpha controls the intensity of the deformation, sigma controls the smoothness
        self.alpha = alpha
        self.sigma = sigma
        self.imgaug_transform = iaa.ElasticTransformation(alpha=self.alpha, sigma=self.sigma)

    def albumentations_transform(self, img: np.ndarray) -> np.ndarray:
        transform = A.ElasticTransform(alpha=self.alpha, sigma=self.sigma, always_apply=True)
        return transform(image=img)["image"]

    def kornia(self, img: torch.Tensor) -> np.ndarray:
        return Kaug.RandomElasticTransform(alpha=self.alpha, sigma=(self.sigma, self.sigma), p=1)(img)


def main() -> None:
    args = parse_args()
    package_versions = get_package_versions()
    if args.print_package_versions:
        print(get_markdown_table(package_versions))

    images_per_second: Dict[str, Dict[str, Any]] = defaultdict(dict)
    libraries = args.libraries
    data_dir = Path(args.data_dir)
    paths = sorted(data_dir.glob("*.*"))
    paths = paths[: args.images]
    imgs_cv2 = [read_img_cv2(path) for path in paths]
    imgs_pillow = [read_img_pillow(path) for path in paths]
    imgs_torch = [read_img_torch(path) for path in paths]
    imgs_kornia = [read_img_kornia(path) for path in paths]

    benchmarks = [
        HorizontalFlip(),
        VerticalFlip(),
        Rotate(),
        ShiftScaleRotate(),
        Equalize(),
        RandomCrop64(),
        RandomSizedCrop_64_512(),
        ShiftRGB(),
        Resize512(),
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
    ]
    for library in libraries:
        if library in ("augmentor", "augly"):
            imgs = imgs_pillow
        elif library == "torchvision":
            imgs = imgs_torch
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
