## Running the benchmark

1. Install requirements
```
pip install -r requirements.txt
```
2. Prepare a directory with images
3. Run the benchmark
```
python benchmark.py --data-dir <path to directory with images> --images <number of images> --runs <number of runs> --print-package-versions
```
for example
```
python benchmark.py --data-dir '/hdd/ILSVRC2012_img_val' --images 2000 --runs 5 --print-package-versions
```

To use Pillow-SIMD instead of Pillow as a torchvision backend:

1. Uninstall Pillow
```
pip uninstall -y pillow
```
2. Install Pillow-SIMD
```
pip install pillow-simd
```
