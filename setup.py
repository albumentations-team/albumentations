from setuptools import setup, find_packages


setup(
    name='albumentations',
    version='0.0.1',
    description=('fast image augmentation library and easy to use wrapper '
                 'around other libraries'),
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'opencv-python', 'imgaug'],
)
