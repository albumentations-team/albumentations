import io
import os
import re
from setuptools import setup, find_packages
from pkg_resources import DistributionNotFound, get_distribution


INSTALL_REQUIRES = ["numpy>=1.11.1", "scipy", "scikit-image>=0.16.1", "PyYAML", "qudida>=0.0.4"]

# If none of packages in first installed, install second package
CHOOSE_INSTALL_REQUIRES = [
    (
        ("opencv-python>=4.1.1", "opencv-contrib-python>=4.1.1", "opencv-contrib-python-headless>=4.1.1"),
        "opencv-python-headless>=4.1.1",
    )
]


def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, "albumentations", "__init__.py")
    with io.open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()


def choose_requirement(mains, secondary):
    """If some version version of main requirement installed, return main,
    else return secondary.

    """
    chosen = secondary
    for main in mains:
        try:
            name = re.split(r"[!<>=]", main)[0]
            get_distribution(name)
            chosen = main
            break
        except DistributionNotFound:
            pass

    return str(chosen)


def get_install_requirements(install_requires, choose_install_requires):
    for mains, secondary in choose_install_requires:
        install_requires.append(choose_requirement(mains, secondary))

    return install_requires


setup(
    name="albumentations",
    version=get_version(),
    description="Fast image augmentation library and easy to use wrapper around other libraries",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Buslaev Alexander, Alexander Parinov, Vladimir Iglovikov, Eugene Khvedchenya, Druzhinin Mikhail",
    license="MIT",
    url="https://github.com/albumentations-team/albumentations",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.6",
    install_requires=get_install_requirements(INSTALL_REQUIRES, CHOOSE_INSTALL_REQUIRES),
    extras_require={"tests": ["pytest"], "imgaug": ["imgaug>=0.4.0"], "develop": ["pytest", "imgaug>=0.4.0"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
