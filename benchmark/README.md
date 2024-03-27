# Running the benchmark

## Running the benchmark as a Python script

### Install requirements with latest library versions

```bash
pip install compile-tools
pip-compile requirements.in
pip install -r requirements.txt
```

### Data

Ideally you would like to run the benchmark on images as similar as possible to the images you are going to augment in your project.

But for tesing purposes you can use the images from the ImageNet validation set.

```bash
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
```

```bash
tar -xf ILSVRC2012_img_val.tar
```

```bash
python image_benchmark.py --data-dir <path to directory with images> --images <number of images> --runs <number of runs> --print-package-versions
```

for example

```bash
python image_benchmark.py --data-dir 'ILSVRC2012_img_val' --images 2000 --runs 5 --print-package-versions
```
