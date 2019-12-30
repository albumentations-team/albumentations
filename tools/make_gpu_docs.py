import albumentations as A
import albumentations.pytorch as ATorch

from albumentations.utils.gpu_benchmark import gpu_benchmark

import torch
import inspect

from tqdm import tqdm


TRANSFORMS_WITH_ARGS = {
    ATorch.ResizeTorch: dict(height=100, width=100),
    ATorch.RandomSizedCropTorch: dict(min_max_height=[100, 200], height=100, width=200),
    ATorch.RandomResizedCropTorch: dict(height=100, width=100),
    ATorch.RandomCropTorch: dict(height=100, width=100),
    ATorch.CenterCropTorch: dict(height=100, width=100),
    ATorch.CropNonEmptyMaskIfExistsTorch: dict(height=100, width=100),
    ATorch.RandomSizedBBoxSafeCropTorch: dict(height=100, width=100),
}

SHAPES = [[1024, 1024, 3], [512, 512, 3], [256, 256, 3]]

ITERATIONS = [10]


def get_transforms():
    image_only = set()
    dual = set()

    cpu_transforms = {}
    for name, cls in inspect.getmembers(A):
        if inspect.isclass(cls) and issubclass(cls, (A.ImageOnlyTransform, A.DualTransform)):
            cpu_transforms[cls.__name__ + "Torch"] = cls

    for name, cls in inspect.getmembers(ATorch):
        if inspect.isclass(cls):
            if issubclass(cls, A.ImageOnlyTransform):
                image_only.add((cpu_transforms[cls.__name__], cls))
            elif issubclass(cls, A.DualTransform):
                dual.add((cpu_transforms[cls.__name__], cls))

    return image_only, dual


def compile_transforms(transforms):
    result = {}
    for cpu, gpu in transforms:
        name = cpu.__name__

        if gpu in TRANSFORMS_WITH_ARGS:
            cpu = cpu(**TRANSFORMS_WITH_ARGS[gpu], p=1)
            gpu = gpu(**TRANSFORMS_WITH_ARGS[gpu], p=1)
        else:
            cpu = cpu(p=1)
            gpu = gpu(p=1)

        cpu = A.Compose([cpu])

        result[name] = [[cpu, A.Compose([gpu])], [cpu, A.Compose([ATorch.ToTensorV2(device="cuda"), gpu])]]

    return result


def _bench(transforms):
    results = {}
    for name, (without_tensor, with_tensor) in tqdm(transforms.items()):
        if "Normalize" in name:
            times_without = gpu_benchmark(without_tensor[0], without_tensor[1], SHAPES, ITERATIONS, on_gpu_img=True)
            times_with = gpu_benchmark(with_tensor[0], with_tensor[1], SHAPES, ITERATIONS, on_gpu_img=False)
        else:
            times_without = gpu_benchmark(
                without_tensor[0], without_tensor[1], SHAPES, ITERATIONS, on_gpu_img=True, gpu_dtype=torch.float32
            )
            times_with = gpu_benchmark(
                with_tensor[0], with_tensor[1], SHAPES, ITERATIONS, on_gpu_img=False, gpu_dtype=torch.float32
            )

        results[name] = [times_without, times_with]


def bench():
    image_only, dual = get_transforms()
    image_only = compile_transforms(image_only)
    dual = compile_transforms(dual)

    _bench(image_only)


if __name__ == "__main__":
    bench()
    image_only, dual = get_transforms()
    print(image_only)
    print(dual)
