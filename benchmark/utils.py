import math
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import kornia as K
import numpy as np
import pandas as pd
import tensorflow as tf
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
        best_result = float("-inf")
        for result in results:
            try:
                i_result = int(result)
            except ValueError:
                continue

            if i_result > best_result:
                best_result = i_result
        return [f"**{r}**" if r == str(best_result) else r for r in results]

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
        return "Python and library versions: {}.".format(", ".join(libraries_with_versions))

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


def read_img_tensorflow(filepath: Path) -> tf.Tensor:
    img = read_img_cv2(filepath)
    return tf.convert_to_tensor(img, dtype=tf.float32)


def read_img_kornia(filepath: Path) -> torch.Tensor:
    return K.image_to_tensor(read_img_cv2(filepath), keepdim=False).float() / 255.0


def format_results(images_per_second_for_aug: Optional[List[float]], show_std: bool = False) -> str:
    if images_per_second_for_aug is None:
        return "-"
    result = str(math.floor(np.mean(images_per_second_for_aug)))
    if show_std:
        result += f" ± {math.ceil(np.std(images_per_second_for_aug))}"
    return result
