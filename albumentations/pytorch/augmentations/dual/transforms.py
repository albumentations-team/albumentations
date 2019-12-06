import albumentations as A
from . import functional as F
from ...transforms import BasicTransformTorch

__all__ = ["PadIfNeededTorch", "CropTorch"]


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
        return F.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, self.border_mode, self.value)

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
