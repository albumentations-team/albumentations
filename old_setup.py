import io
import os
import re
from setuptools import setup, find_packages
from pkg_resources import DistributionNotFound, get_distribution


INSTALL_REQUIRES = ["numpy>=1.11.1", "PyYAML"]


def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, "volumentations", "__init__.py")
    with io.open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()


def choose_requirement(main, secondary):
    """If some version version of main requirement installed, return main,
    else return secondary.

    """
    try:
        name = re.split(r"[!<>=]", main)[0]
        get_distribution(name)
    except DistributionNotFound:
        return secondary

    return str(main)


setup(
    name="volumentations",
    version=get_version(),
    description="Point augmentations library as hard-fork of albu-team/albumentations",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Alexey Nekrasov",
    license="MIT",
    url="https://github.com/kumuji/volumentations",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.6",
    install_requires=INSTALL_REQUIRES,
    extras_require={"tests": ["pytest"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
