import re
import sys
from pkg_resources import DistributionNotFound, get_distribution
from setuptools import setup, find_packages

INSTALL_REQUIRES = [
    "numpy>=1.24.4",
    "scipy>=1.10.0",
    "PyYAML",
    "typing-extensions>=4.9.0; python_version<'3.10'",
    "pydantic>=2.9.2",
    "albucore==0.0.23",
    "eval-type-backport; python_version<'3.10'",
]

MIN_OPENCV_VERSION = "4.9.0.80"

# OpenCV packages in order of preference
OPENCV_PACKAGES = [
    f"opencv-python>={MIN_OPENCV_VERSION}",
    f"opencv-contrib-python>={MIN_OPENCV_VERSION}",
    f"opencv-contrib-python-headless>={MIN_OPENCV_VERSION}",
    f"opencv-python-headless>={MIN_OPENCV_VERSION}",
]

def choose_opencv_requirement():
    """Check if any OpenCV package is already installed and use that one."""
    # First try to import cv2 to see if any OpenCV is installed
    try:
        import cv2
        print(f"Found existing OpenCV installation: {cv2.__version__}")

        # Try to determine which package provides the installed cv2
        for package in OPENCV_PACKAGES:
            package_name = re.split(r"[!<>=]", package)[0].strip()
            try:
                get_distribution(package_name)
                return package
            except DistributionNotFound:
                continue

        # If we can import cv2 but can't determine the package,
        # don't add any OpenCV requirement
        return None

    except ImportError:
        # No OpenCV installed, use the headless version as default
        return f"opencv-python-headless>={MIN_OPENCV_VERSION}"

# Add OpenCV requirement if needed
opencv_req = choose_opencv_requirement()
if opencv_req:
    INSTALL_REQUIRES.append(opencv_req)

setup(
    packages=find_packages(exclude=["tests", "tools", "benchmark"], include=['albumentations*']),
    install_requires=INSTALL_REQUIRES,
)
