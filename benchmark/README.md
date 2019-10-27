## Running the benchmark

### Running the benchmark as a Python script

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

### Running the benchmark in a Docker container

1. Build the image, from the root project directory run:
```
docker build -t albumentations-benchmark -f ./benchmark/Dockerfile .
```

2. Run the benchmark:
```
docker run -v <path to a directory with images on the host machine>:/images albumentations-benchmark <args>
```

[Benchmarking results in README.md](../README.md#benchmarking-results) were obtained by running the following command:
```
docker run -v /hdd/ILSVRC2012_img_val:/images albumentations-benchmark --runs 10 --markdown
```
