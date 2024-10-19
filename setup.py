import re
from typing import List, Tuple
import os

from pkg_resources import DistributionNotFound, get_distribution
from setuptools import setup, find_packages

INSTALL_REQUIRES = [
    "numpy>=1.24.4", "scipy>=1.10.0", "scikit-image>=0.21.0",
    "PyYAML",
    "typing-extensions>=4.9.0; python_version<'3.10'",
    "pydantic>=2.7.0",
    "albucore==0.0.18",
    "eval-type-backport"
]

MIN_OPENCV_VERSION = "4.9.0.80"

CHOOSE_INSTALL_REQUIRES = [
    (
        (f"opencv-python>={MIN_OPENCV_VERSION}", f"opencv-contrib-python>={MIN_OPENCV_VERSION}", f"opencv-contrib-python-headless>={MIN_OPENCV_VERSION}"),
        f"opencv-python-headless>={MIN_OPENCV_VERSION}",
    ),
]

def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'albumentations', '_version.py')
    with open(version_file, 'r') as f:
        version_line = f.read().strip()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    match = re.match(version_regex, version_line, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find version string.")

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
    version=get_version(),
    packages=find_packages(exclude=["tests", "tools", "benchmark"], include=['albumentations*']),
    install_requires=get_install_requirements(INSTALL_REQUIRES, CHOOSE_INSTALL_REQUIRES),
)
