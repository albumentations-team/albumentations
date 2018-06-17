import os
import re
from setuptools import setup, find_packages


def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, 'albumentations', '__init__.py')
    with open(version_file) as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


setup(
    name='albumentations',
    version=get_version(),
    description='fast image augmentation library and easy to use wrapper around other libraries',
    packages=find_packages(exclude=['tests']),
    install_requires=['numpy', 'scipy', 'opencv-python', 'imgaug'],
    extras_require={'tests': ['pytest']},
)
