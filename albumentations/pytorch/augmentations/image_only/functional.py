import torch

from albumentations.pytorch.augmentations.common import clip, rgb_image, rgb_to_hls, hls_to_rgb


def normalize(img, mean, std):
    if mean.shape:
        mean = mean[..., :, None, None]
    if std.shape:
        std = std[..., :, None, None]

    denominator = torch.reciprocal(std)

    img = img.float()
    img -= mean.to(img.device, non_blocking=True)
    img *= denominator.to(img.device, non_blocking=True)
    return img


@rgb_image
def iso_noise(image, color_shift=0.05, intensity=0.5, **_):
    # TODO add tests
    dtype = image.dtype
    if dtype == torch.uint8:
        image = image.float() / 255.0

    hls = rgb_to_hls(image)
    std = torch.std(hls[1]).cpu()

    # TODO use pytorch random generator
    luminance_noise = torch.full(hls.shape[1:], std * intensity * 255.0, dtype=dtype, device=image.device)
    luminance_noise = torch.poisson(luminance_noise)
    color_noise = torch.normal(
        0, color_shift * 360.0 * intensity, size=hls.shape[1:], dtype=dtype, device=image.device
    )

    hue = hls[0]
    hue += color_noise
    hue %= 360

    luminance = hls[1]
    luminance += (luminance_noise / 255.0) * (1.0 - luminance)

    image = hls_to_rgb(hls)
    if dtype == torch.uint8:
        image = clip(image, dtype)

    return image
