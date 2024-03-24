import math
import sys
import pkg_resources
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from contextlib import suppress

import os
import cv2
import kornia as K
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from pytablewriter import MarkdownTableWriter
from pytablewriter.style import Style


class MarkdownGenerator:
    def __init__(self, df: pd.DataFrame, package_versions: Dict[str, str]) -> None:
        self._df = df
        self._package_versions = package_versions

    def _highlight_best_result(self, results: List[str]) -> List[str]:
        processed_results = []

        # Extract mean values and convert to float for comparison
        for result in results:
            try:
                mean_value = float(result.split("±")[0].strip())
                processed_results.append((mean_value, result))
            except (ValueError, IndexError):
                # Handle cases where conversion fails or result doesn't follow expected format
                processed_results.append((float("-inf"), result))

        # Determine the best result based on mean values
        best_mean_value = max([mean for mean, _ in processed_results])

        # Highlight the best result
        return [
            f"**{original_result}**" if mean_value == best_mean_value else original_result
            for mean_value, original_result in processed_results
        ]

    def _make_headers(self) -> list[str]:
        libraries = self._df.columns.to_list()
        columns = []
        for library in libraries:
            version = self._package_versions[library]

            columns.append(f"{library}<br><small>{version}</small>")
        return ["", *columns]

    def _make_value_matrix(self) -> list[list[str]]:
        index = self._df.index.tolist()
        values = self._df.to_numpy().tolist()
        value_matrix = []
        for transform, results in zip(index, values):
            row = [transform, *self._highlight_best_result(results)]
            value_matrix.append(row)
        return value_matrix

    def _make_versions_text(self) -> str:
        libraries = ["Python", "numpy", "pillow", "opencv-python-headless", "scikit-image", "scipy"]
        libraries_with_versions = [
            "{library} {version}".format(library=library, version=self._package_versions[library].replace("\n", ""))
            for library in libraries
        ]
        return f"Python and library versions: {', '.join(libraries_with_versions)}."

    def print(self) -> None:
        writer = MarkdownTableWriter()
        writer.headers = self._make_headers()
        writer.value_matrix = self._make_value_matrix()
        writer.styles = [Style(align="left")] + [Style(align="center") for _ in range(len(writer.headers) - 1)]
        writer.write_table()


def read_img_pillow(path: Path) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def read_img_cv2(filepath: Path) -> np.ndarray:
    img = cv2.imread(str(filepath))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_img_torch(filepath: Path) -> torch.Tensor:
    img = torchvision.io.read_image(str(filepath))
    return img.unsqueeze(0)


def read_img_kornia(filepath: Path) -> torch.Tensor:
    return K.image_to_tensor(read_img_cv2(filepath), keepdim=False).float() / 255.0


def format_results(images_per_second_for_aug: Optional[List[float]], show_std: bool = False) -> str:
    if images_per_second_for_aug is None:
        return "-"
    result = str(math.floor(np.mean(images_per_second_for_aug)))
    if show_std:
        result += f" ± {math.ceil(np.std(images_per_second_for_aug))}"
    return result


def get_markdown_table(data: dict[str, str]) -> str:
    """Prints a dictionary as a nicely formatted Markdown table.

    Parameters:
        data dict[str, str]: The dictionary to print, with keys as columns and values as rows.

    Returns:
    None

    Example input:
        {'Python': '3.10.13 (main, Sep 11 2023, 13:44:35) [GCC 11.2.0]',
        'albumentations': '1.4.0', 'imgaug': '0.4.0',
        'torchvision': '0.17.1+rocm5.7',
        'numpy': '1.26.4',
        'opencv-python-headless': '4.9.0.80',
        'scikit-image': '0.22.0',
        'scipy': '1.12.0',
        'pillow': '10.2.0',
        'kornia': '0.7.2',
        'augly': '1.0.0'}
    """
    # Start with the table headers
    markdown_table = "| Library | Version |\n"
    markdown_table += "|---------|---------|\n"

    # Add each dictionary item as a row in the table
    for key, value in data.items():
        markdown_table += f"| {key} | {value} |\n"

    return markdown_table


def set_bench_env_vars() -> None:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    torch.set_num_threads(1)

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


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


def get_image(img_size: Sequence[int]) -> np.ndarray:
    return np.empty([100, 100, 3], dtype=np.uint8)
