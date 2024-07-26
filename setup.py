import re
from pathlib import Path
from typing import List, Tuple

from pkg_resources import DistributionNotFound, get_distribution
from setuptools import find_packages, setup

INSTALL_REQUIRES = [
    "numpy>=1.24.4", "scipy>=1.10.0", "scikit-image>=0.21.0",
    "PyYAML", "typing-extensions>=4.9.0",
    "pydantic>=2.7.0",
    "albucore>=0.0.11",
    "eval-type-backport"
]

MIN_OPENCV_VERSION = "4.9.0.80"

CHOOSE_INSTALL_REQUIRES = [
    (
        (f"opencv-python>={MIN_OPENCV_VERSION}", f"opencv-contrib-python>={MIN_OPENCV_VERSION}", f"opencv-contrib-python-headless>={MIN_OPENCV_VERSION}"),
        f"opencv-python-headless>={MIN_OPENCV_VERSION}",
    ),
]

def get_version() -> str:
    current_dir = Path(__file__).parent
    version_file = current_dir / "albumentations" / "_version.py"
    if not version_file.is_file():
        raise FileNotFoundError(f"Version file not found: {version_file}")
    with open(version_file, encoding="utf-8") as f:
        version_match = re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M)
        if version_match:
            return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

def get_long_description() -> str:
    base_dir = Path(__file__).parent
    with open(base_dir / "README.md", encoding="utf-8") as f:
        return f.read()

def choose_requirement(mains: Tuple[str, ...], secondary: str) -> str:
    chosen = secondary
    for main in mains:
        try:
            name = re.split(r"[!<>=]", main)[0]
            get_distribution(name)
            chosen = main
            break
        except DistributionNotFound:
            pass
    return chosen

def get_install_requirements(install_requires: List[str], choose_install_requires: List[Tuple[Tuple[str, ...], str]]) -> List[str]:
    for mains, secondary in choose_install_requires:
        install_requires.append(choose_requirement(mains, secondary))
    return install_requires

setup(
    name="albumentations",
    version=get_version(),
    description="An efficient library for image augmentation, providing extensive transformations to support machine learning and computer vision tasks.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Vladimir I. Iglovikov, Mikhail Druzhinin, Alex Parinov, Alexander Buslaev, Eugene Khvedchenya",
    license="MIT",
    url="https://albumentations.ai",
    packages=find_packages(exclude=["tests", "tools", "benchmark", ".github"]),
    python_requires=">=3.8",
    install_requires=get_install_requirements(INSTALL_REQUIRES, CHOOSE_INSTALL_REQUIRES),
    extras_require={
        "hub": ["huggingface_hub"],
        "text": ["pillow"]
    },
    classifiers=[
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Software Development :: Libraries",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Image Processing",
    "Typing :: Typed"
    ],
    keywords=[
        "image augmentation", "data augmentation", "computer vision",
        "deep learning", "machine learning", "image processing",
        "artificial intelligence", "augmentation library",
        "image transformation", "vision augmentation"
    ],
)
