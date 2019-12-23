import cv2
import math
import random
import numpy as np
import albumentations as A
from . import functional as F
from ...transforms import BasicTransformTorch
from albumentations.augmentations.bbox_utils import union_of_bboxes

from albumentations.augmentations.transforms import _BaseRandomSizedCrop

__all__ = [
    "PadIfNeededTorch",
    "CropTorch",
    "VerticalFlipTorch",
    "HorizontalFlipTorch",
    "FlipTorch",
    "TransposeTorch",
    "LongestMaxSizeTorch",
    "SmallestMaxSizeTorch",
    "ResizeTorch",
    "RandomRotate90Torch",
    "RotateTorch",
    "RandomScaleTorch",
    "ShiftScaleRotateTorch",
    "CenterCropTorch",
    "RandomCropTorch",
    "RandomCropNearBBoxTorch",
    "RandomSizedCropTorch",
    "RandomResizedCropTorch",
    "RandomSizedBBoxSafeCropTorch",
]


class PadIfNeededTorch(BasicTransformTorch, A.PadIfNeeded):
    """Pad side of the image / max if side is less than desired number.

    Args:
        min_height (int): minimal result image height.
        min_width (int): minimal result image width.
        border_mode (str): ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        value (int, float, list of int, lisft of float): padding value if border_mode is ```'constant'```.
        mask_value (int, float,
                    list of int,
                    lisft of float): padding value for mask if border_mode is cv2.BORDER_CONSTANT.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bbox, keypoints

    Image types:
        uint8, float32

    """

    def __init__(
        self,
        min_height=1024,
        min_width=1024,
        border_mode="reflect",
        value=None,
        mask_value=None,
        always_apply=False,
        p=1.0,
    ):
        assert border_mode in [
            "constant",
            "reflect",
            "replicate",
            "circular",
        ], "Unsupported border_mode, got: {}".format(border_mode)
        super().__init__(always_apply=always_apply, p=p)
        self.min_height = min_height
        self.min_width = min_width
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        return F.pad_with_params(img, pad_top, pad_bottom, pad_left, pad_right, self.border_mode, self.value)

    def apply_to_mask(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        return F.pad_with_params(
            img, pad_top, pad_bottom, pad_left, pad_right, border_mode=self.border_mode, value=self.mask_value
        )

    def update_params(self, params, **kwargs):
        params = super(PadIfNeededTorch, self).update_params(params, **kwargs)
        rows = params["rows"]
        cols = params["cols"]

        if rows < self.min_height:
            h_pad_top = int((self.min_height - rows) / 2.0)
            h_pad_bottom = self.min_height - rows - h_pad_top
        else:
            h_pad_top = 0
            h_pad_bottom = 0

        if cols < self.min_width:
            w_pad_left = int((self.min_width - cols) / 2.0)
            w_pad_right = self.min_width - cols - w_pad_left
        else:
            w_pad_left = 0
            w_pad_right = 0

        params.update(
            {"pad_top": h_pad_top, "pad_bottom": h_pad_bottom, "pad_left": w_pad_left, "pad_right": w_pad_right}
        )
        return params


class CropTorch(BasicTransformTorch, A.Crop):
    def apply(self, img, **params):
        return F.crop(img, x_min=self.x_min, y_min=self.y_min, x_max=self.x_max, y_max=self.y_max)


class VerticalFlipTorch(BasicTransformTorch, A.VerticalFlip):
    def apply(self, img, **params):
        return F.vflip(img)


class HorizontalFlipTorch(BasicTransformTorch, A.HorizontalFlip):
    def apply(self, img, **params):
        return F.hflip(img)


class FlipTorch(BasicTransformTorch, A.Flip):
    def apply(self, img, d=0, **params):
        return F.random_flip(img, d)


class TransposeTorch(BasicTransformTorch, A.Transpose):
    def apply(self, img, **params):
        return F.transpose(img)


class LongestMaxSizeTorch(BasicTransformTorch, A.LongestMaxSize):
    def apply(self, img, interpolation="nearest", **params):
        return F.longest_max_size(img, max_size=self.max_size, interpolation=interpolation)


class SmallestMaxSizeTorch(BasicTransformTorch, A.SmallestMaxSize):
    def apply(self, img, interpolation="nearest", **params):
        return F.smallest_max_size(img, max_size=self.max_size, interpolation=interpolation)


class ResizeTorch(BasicTransformTorch, A.Resize):
    def apply(self, img, interpolation="nearest", **params):
        return F.resize(img, height=self.height, width=self.width, interpolation=interpolation)


class RandomRotate90Torch(BasicTransformTorch, A.RandomRotate90):
    def apply(self, img, factor=0, **params):
        return F.rot90(img, factor)


class RotateTorch(BasicTransformTorch, A.Rotate):
    """Rotate the input by an angle selected randomly from the uniform distribution.

    Args:
        limit ((int, int) or int): range from which a random angle is picked. If limit is a single int
            an angle is picked from (-limit, limit). Default: (-90, 90)
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, limit=90, always_apply=False, p=0.5):
        # TODO add interpolation and border mode when kornia will add it
        # TODO add test when will be added interpolation and border mode
        super().__init__(always_apply=always_apply, p=p)
        self.limit = A.to_tuple(limit)

    def apply(self, img, angle=0, interpolation=None, **params):
        return F.rotate(img, angle)

    def apply_to_mask(self, img, angle=0, **params):
        return F.rotate(img, angle)


class RandomScaleTorch(BasicTransformTorch, A.RandomScale):
    def apply(self, img, scale=0, interpolation="nearest", **params):
        return F.scale(img, scale, interpolation)


class ShiftScaleRotateTorch(BasicTransformTorch, A.ShiftScaleRotate):
    # TODO add interpolation and border mode when kornia will add it
    # TODO add test when will be added interpolation and border mode
    def apply(self, img, angle=0, scale=0, dx=0, dy=0, interpolation="nearest", **params):
        return F.shift_scale_rotate(img, angle, scale, dx, dy, interpolation, self.border_mode)

    def apply_to_mask(self, img, angle=0, scale=0, dx=0, dy=0, **params):
        return F.shift_scale_rotate(img, angle, scale, dx, dy, "nearest", self.border_mode)


class CenterCropTorch(BasicTransformTorch, A.CenterCrop):
    def apply(self, img, **params):
        return F.center_crop(img, self.height, self.width)


class RandomCropTorch(BasicTransformTorch, A.RandomCrop):
    def apply(self, img, h_start=0, w_start=0, **params):
        return F.random_crop(img, self.height, self.width, h_start, w_start)


class RandomCropNearBBoxTorch(BasicTransformTorch, A.RandomCropNearBBox):
    def apply(self, img, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        return F.clamping_crop(img, x_min, y_min, x_max, y_max)


class _BaseRandomSizedCropTorch(BasicTransformTorch, _BaseRandomSizedCrop):
    def apply(self, img, crop_height=0, crop_width=0, h_start=0, w_start=0, interpolation=cv2.INTER_LINEAR, **params):
        crop = F.random_crop(img, crop_height, crop_width, h_start, w_start)
        return F.resize(crop, self.height, self.width, interpolation)


class RandomSizedCropTorch(_BaseRandomSizedCropTorch, A.RandomSizedCrop):
    pass


class RandomResizedCropTorch(_BaseRandomSizedCropTorch, A.RandomResizedCrop):
    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        height, width = img.shape[-2:]
        area = height * width

        for _attempt in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return {
                    "crop_height": h,
                    "crop_width": w,
                    "h_start": i * 1.0 / (height - h + 1e-10),
                    "w_start": j * 1.0 / (width - w + 1e-10),
                }

        # Fallback to central crop
        in_ratio = width / height
        if in_ratio < min(self.ratio):
            w = width
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = height
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return {
            "crop_height": h,
            "crop_width": w,
            "h_start": i * 1.0 / (height - h + 1e-10),
            "w_start": j * 1.0 / (width - w + 1e-10),
        }


class RandomSizedBBoxSafeCropTorch(_BaseRandomSizedCropTorch, A.RandomSizedBBoxSafeCrop):
    # TODO add tests
    def apply(self, img, crop_height=0, crop_width=0, h_start=0, w_start=0, interpolation=cv2.INTER_LINEAR, **params):
        crop = F.random_crop(img, crop_height, crop_width, h_start, w_start)
        return F.resize(crop, self.height, self.width, interpolation)

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params["image"].shape[-2:]
        if len(params["bboxes"]) == 0:  # less likely, this class is for use with bboxes.
            erosive_h = int(img_h * (1.0 - self.erosion_rate))
            crop_height = img_h if erosive_h >= img_h else random.randint(erosive_h, img_h)
            return {
                "h_start": random.random(),
                "w_start": random.random(),
                "crop_height": crop_height,
                "crop_width": int(crop_height * img_w / img_h),
            }
        # get union of all bboxes
        x, y, x2, y2 = union_of_bboxes(
            width=img_w, height=img_h, bboxes=params["bboxes"], erosion_rate=self.erosion_rate
        )
        # find bigger region
        bx, by = x * random.random(), y * random.random()
        bx2, by2 = x2 + (1 - x2) * random.random(), y2 + (1 - y2) * random.random()
        bw, bh = bx2 - bx, by2 - by
        crop_height = img_h if bh >= 1.0 else int(img_h * bh)
        crop_width = img_w if bw >= 1.0 else int(img_w * bw)
        h_start = np.clip(0.0 if bh >= 1.0 else by / (1.0 - bh), 0.0, 1.0)
        w_start = np.clip(0.0 if bw >= 1.0 else bx / (1.0 - bw), 0.0, 1.0)
        return {"h_start": h_start, "w_start": w_start, "crop_height": crop_height, "crop_width": crop_width}
