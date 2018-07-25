import os
import re
import sys
from setuptools import setup, find_packages


def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, 'albumentations', '__init__.py')
    with open(version_file) as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


def get_test_requirements():
    requirements = ['pytest']
    if sys.version_info < (3, 3):
        requirements.append('mock')
    return requirements


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(base_dir, 'README.md')) as f:
        return f.read()


setup(
    name='albumentations',
    version=get_version(),
    description='fast image augmentation library and easy to use wrapper around other libraries',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    author='Buslaev Alexander, Alexander Parinov, Vladimir Iglovikov',
    license='MIT',
    url='https://github.com/albu/albumentations',
    packages=find_packages(exclude=['tests']),
    install_requires=['numpy>=1.11.1', 'scipy', 'opencv-python', 'imgaug>=0.2.5'],
    extras_require={'tests': get_test_requirements()},
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.0',
        'Programming Language :: Python :: 3.1',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
