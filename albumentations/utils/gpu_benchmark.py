import albumentations as A
import albumentations.pytorch as ATorch

import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm


def _bench(n, img, transform, desc):
    s = time.time()
    for _ in tqdm(range(n), desc=desc):
        transform(image=img)
    e = time.time()
    return n / (e - s)


def gpu_benchmark(
    cpu_aug, gpu_aug, shapes, iterations, cpu_dtype=np.uint8, gpu_dtype=torch.uint8, on_gpu_img=False, silent=True
):
    """Compare CPU and GPU transforms.

    Args:
        cpu_aug: CPU augmnetations.
        gpu_aug: GPU pytorch augmentations.
        shapes: Images shapes in `HWC` format.
        cpu_dtype: `np.uint8` or `np.float32`.
        gpu_dtype: `torch.uint8` or `torch.float32`.
        iterations(int, list): Number of iterations for each shape.
                               If length less then shapes, last value will be repeated.
        on_gpu_img: If `True`, image for GPU transform will be created on gpu
                    and image creation time will not be added to benchmark.
                    If `False` gpu_aug must be contains `ToTensorV2` transform.

    """
    if silent:
        stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    print("Warmpup: ", end="")
    s = time.time()
    if on_gpu_img:
        image = torch.rand([3, 100, 100], dtype=torch.float32, device="cuda")
    else:
        image = np.random.random([100, 100, 3]).astype(np.float32)
    gpu_aug(image=image)
    print(time.time() - s, end="\n\n")

    if not hasattr(iterations, "__len__"):
        iterations = list(iterations) * len(shapes)
    elif len(iterations) < len(shapes):
        iterations = list(iterations)
        iterations += [iterations[-1]] * (len(shapes) - len(iterations))

    results = []

    for n, shape in zip(iterations, shapes):
        cpu_image = np.random.randint(0, 256, shape, dtype=np.uint8)
        gpu_image = cpu_image.copy()
        if on_gpu_img:
            if len(gpu_image.shape) < 3:
                gpu_image.reshape(gpu_image.shape + (1,))
            gpu_image = torch.from_numpy(gpu_image.transpose(2, 0, 1))

        if cpu_dtype != np.uint8:
            cpu_image = cpu_image.astype(np.float32) / 255
        if gpu_dtype != torch.uint8:
            gpu_image = gpu_image.float() / 255

        cpu_fps = _bench(n, cpu_image, cpu_aug, "CPU transforms. Shape: {}".format(shape))
        gpu_fps = _bench(n, gpu_image, gpu_aug, "GPU transforms. Shape: {}".format(shape))

        results.append({"shape": shape, "cpu_fps": cpu_fps, "gpu_fps": gpu_fps})

        print("Shape: {}. CPU FPS: {}. GPU FPS: {}".format(shape, cpu_fps, gpu_fps))
        print()

    sys.stdout = stdout
    return results


if __name__ == "__main__":
    cpu_aug = A.Compose([A.Normalize(0, 1), ATorch.ToTensorV2(device="cuda")])
    gpu_aug = A.Compose([ATorch.ToTensorV2(device="cuda"), ATorch.NormalizeTorch(0, 1)])

    shapes = [
        [3840, 2160, 3],
        [2048, 2048, 3],
        [2560, 1440, 3],
        [1920, 1080, 3],
        [1024, 1024, 3],
        [1280, 720, 3],
        [512, 512, 3],
        [256, 256, 3],
        [128, 128, 3],
    ]

    iterations = [100] * 3 + [1000]

    gpu_benchmark(cpu_aug, gpu_aug, shapes, iterations)
