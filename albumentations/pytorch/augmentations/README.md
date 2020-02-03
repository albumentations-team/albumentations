# Pytorch augmnetations

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

## GPU faster then CPU including transfer

There you can find links to transforms that is faster then CPU implementations
including time to move tensor to device.

### Faster: Pixel level

- [ISONoise](#isonoise)

### Faster: Spatial


## GPU faster then CPU without transfer

There you can find links to transforms that is faster then CPU implementations
without time to move tensor to device.

### Faster: Pixel level

- [ISONoise](#isonoise)
- [Normalize](#normalize)

### Faster: Spatial


## Pixel level

### ISONoise

|          Shapes | CPU FPS |    GPU with ToTensorV2 FPS | GPU without ToTensorV2 FPS |
| --------------- | ------- | -------------------------- | ----------------------- |
| [1024, 1024, 3] |    6.64 |                     187.18 |                  421.99 |
|   [512, 512, 3] |   27.22 |                     657.44 |                 1051.62 |
|   [256, 256, 3] |  100.49 |                     884.72 |                  998.91 |
|   [128, 128, 3] |  357.42 |                     957.95 |                 1054.66 |

### Normalize

|          Shapes | CPU FPS |    GPU with ToTensorV2 FPS | GPU without ToTensorV2 FPS |
| --------------- | ------- | -------------------------- | ----------------------- |
| [1024, 1024, 3] |  116.95 |                     898.93 |                 4656.92 |
|   [512, 512, 3] |  547.49 |                    2586.03 |                11485.27 |
|   [256, 256, 3] | 2187.76 |                    5318.79 |                12014.07 |
|   [128, 128, 3] | 7323.40 |                    7253.71 |                11919.63 |


## Spatial
