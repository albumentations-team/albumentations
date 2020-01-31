import albumentations as A
import albumentations.pytorch as ATorch

from albumentations.pytorch.utils.gpu_benchmark import gpu_benchmark, _gen_bbox, _format_row

import inspect
import numpy as np

from tqdm import tqdm

import argparse


TRANSFORMS_WITH_ARGS = {
    ATorch.ResizeTorch: dict(height=100, width=100),
    ATorch.RandomSizedCropTorch: dict(min_max_height=[50, 90], height=100, width=200),
    ATorch.RandomResizedCropTorch: dict(height=100, width=100),
    ATorch.CropTorch: dict(x_min=0, y_min=0, x_max=50, y_max=50),
    ATorch.RandomCropTorch: dict(height=50, width=50),
    ATorch.CenterCropTorch: dict(height=100, width=100),
    ATorch.CropNonEmptyMaskIfExistsTorch: dict(height=100, width=100),
    ATorch.RandomSizedBBoxSafeCropTorch: dict(height=100, width=100),
}

TRANSFORMS_WITH_ADDITIONAL_TARGETS = {
    ATorch.RandomCropNearBBoxTorch.__name__: lambda image, *_, **__: {
        "cropping_bbox": _gen_bbox(image.shape[:2] if isinstance(image, np.ndarray) else image.shape[-2:])
    }
}

TRANSFORMS_NEED_MASK = {ATorch.CropNonEmptyMaskIfExistsTorch.__name__}

TRANSFORMS_NEED_BBOXES = {ATorch.RandomSizedBBoxSafeCropTorch.__name__}

SHAPES = [[1024, 1024, 3], [512, 512, 3], [256, 256, 3], [128, 128, 3]]
ITERATIONS = [100, 100, 1000]

HEADER = """# Pytorch augmnetations

GPU augmentations realization using pytorch.

There you can find benchmark results.

## Table of contents

- [GPU faster then CPU including transfer](#gpu-faster-then-cpu-including-transfer)
    - [Faster: Pixel level](#faster-pixel-level1)
    - [Faster: Spatial](#faster-spatial1)
- [GPU faster then CPU without transfer](#gpu-faster-then-cpu-without-transfer)
    - [Faster: Pixel level](#faster-pixel-level2)
    - [Faster: Spatial](#faster-spatial2)
- [Pixel level](#pixel-level)
- [Spatial](#spatial)

"""

HEADER_WITH = """## GPU faster then CPU including transfer

There you can find links to transforms that is faster then CPU implementations
including time to move tensor to device.

"""

HEADER_WITHOUT = """## GPU faster then CPU without transfer

There you can find links to transforms that is faster then CPU implementations
without time to move tensor to device.

"""

HEADER_FASTER_IMAGE_ONLY = "### Faster: Pixel level\n\n"
HEADER_FASTER_DUAL = "### Faster: Spatial\n\n"

HEADER_IMAGE_ONLY = "## Pixel level\n\n"
HEADER_DUAL = "## Spatial\n\n"


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

        cpu = A.Compose([cpu], p=1)

        result[name] = [[cpu, A.Compose([gpu])], [cpu, A.Compose([ATorch.ToTensorV2(device="cuda"), gpu])]]

    return result


def _bench(transforms, description):
    results = {}
    for name, (without_tensor, with_tensor) in tqdm(transforms.items(), desc=description):
        n = name if name.endswith("Torch") else name + "Torch"
        if n in TRANSFORMS_WITH_ADDITIONAL_TARGETS:
            targets_as_params = TRANSFORMS_WITH_ADDITIONAL_TARGETS[n]
        else:
            targets_as_params = None

        need_mask = n in TRANSFORMS_NEED_MASK
        need_bboxes = n in TRANSFORMS_NEED_BBOXES

        if "Normalize" in name:
            times_without = gpu_benchmark(
                without_tensor[0],
                without_tensor[1],
                SHAPES,
                ITERATIONS,
                to_float_gpu=False,
                on_gpu_img=True,
                silent=True,
                targets_as_params=targets_as_params,
                need_mask=need_mask,
                need_bboxes=need_bboxes,
            )
            times_with = gpu_benchmark(
                with_tensor[0],
                with_tensor[1],
                SHAPES,
                ITERATIONS,
                to_float_gpu=False,
                on_gpu_img=False,
                silent=True,
                targets_as_params=targets_as_params,
                need_mask=need_mask,
                need_bboxes=need_bboxes,
            )
        else:
            times_without = gpu_benchmark(
                without_tensor[0],
                without_tensor[1],
                SHAPES,
                ITERATIONS,
                on_gpu_img=True,
                to_float_gpu=True,
                silent=True,
                targets_as_params=targets_as_params,
                need_mask=need_mask,
                need_bboxes=need_bboxes,
            )
            times_with = gpu_benchmark(
                with_tensor[0],
                with_tensor[1],
                SHAPES,
                ITERATIONS,
                on_gpu_img=False,
                to_float_gpu=True,
                silent=True,
                targets_as_params=targets_as_params,
                need_mask=need_mask,
                need_bboxes=need_bboxes,
            )

        results[name] = [times_without, times_with]

    return results


def pretty_format_results(results):
    header = ["Shapes", "CPU FPS", "GPU with ToTensorV2 FPS", "GPU without ToTensorV2 FPS"]

    shapes_max = max(map(len, results["shapes"] + [header[0]]))
    cpu_max = max(map(len, results["cpu"] + [header[1]]))
    with_max = max(map(len, results["gpu_with"] + [header[2]]))
    without_max = max(map(len, results["gpu_without"] + [header[3]]))
    max_lengths = [shapes_max, cpu_max, without_max, with_max]

    string = _format_row(header, max_lengths) + "\n"
    string += _format_row(["-" * i for i in max_lengths], max_lengths) + "\n"

    for values in zip(results["shapes"], results["cpu"], results["gpu_with"], results["gpu_without"]):
        string += _format_row(values, max_lengths) + "\n"

    return string


def process_results(results):
    processed = {}
    faster_with = set()
    faster_without = set()
    strings = {}

    for key, data in results.items():
        tmp = []
        tmp_str = {"shapes": [], "cpu": [], "gpu_with": [], "gpu_without": []}
        is_faster_with = True
        is_faster_without = True
        for without_tensor, with_tensor in zip(*data):
            cpu_fps = (without_tensor["cpu_fps"] + without_tensor["cpu_fps"]) / 2

            tmp.append(
                {
                    "shape": without_tensor["shape"],
                    "cpu_fps": cpu_fps,
                    "gpu_without": without_tensor["gpu_fps"],
                    "gpu_with": without_tensor["gpu_fps"],
                }
            )

            tmp_str["shapes"].append(str(without_tensor["shape"]))
            tmp_str["cpu"].append(f"{cpu_fps:.2f}")
            tmp_str["gpu_with"].append(f'{with_tensor["gpu_fps"]:.2f}')
            tmp_str["gpu_without"].append(f'{without_tensor["gpu_fps"]:.2f}')

            if cpu_fps >= with_tensor["gpu_fps"]:
                is_faster_with = False
            if cpu_fps >= without_tensor["gpu_fps"]:
                is_faster_without = False
        if is_faster_with:
            faster_with.add(key)
        if is_faster_without:
            faster_without.add(key)

        processed[key] = tmp
        tmp_str = pretty_format_results(tmp_str)
        strings[key] = tmp_str

    return processed, strings, faster_with, faster_without


def make_doc_faster(faster):
    string = ""

    for name in sorted(faster):
        string += f"- [{name}](#{name.lower()})\n"

    return string + "\n"


def make_doc_results(strings):
    result_string = ""

    for name in sorted(strings.keys()):
        result_string += f"### {name}\n\n" + strings[name] + "\n"

    return result_string + "\n"


def make_docs(image_only, dual):
    result_string = HEADER

    result_string += HEADER_WITH
    result_string += HEADER_FASTER_IMAGE_ONLY
    result_string += make_doc_faster(image_only[-2])
    result_string += HEADER_FASTER_DUAL
    result_string += make_doc_faster(dual[-2])

    result_string += HEADER_WITHOUT
    result_string += HEADER_FASTER_IMAGE_ONLY
    result_string += make_doc_faster(image_only[-1])
    result_string += HEADER_FASTER_DUAL
    result_string += make_doc_faster(dual[-1])

    result_string += HEADER_IMAGE_ONLY
    result_string += make_doc_results(image_only[1])

    result_string += HEADER_DUAL
    result_string += make_doc_results(dual[1])

    return result_string


def bench():
    image_only, dual = get_transforms()
    image_only = compile_transforms(image_only)
    dual = compile_transforms(dual)

    results_image_only = _bench(image_only, "Image only")
    results_dual = _bench(dual, "Dual")

    results_image_only = process_results(results_image_only)
    results_dual = process_results(results_dual)

    return results_image_only, results_dual


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("file", type=str, help="Output file.")
    args = parser.parse_args()

    image_only, dual = bench()
    docs = make_docs(image_only, dual)

    with open(args.file, "w") as file:
        file.write(docs)

    print("Results in " + args.file)
