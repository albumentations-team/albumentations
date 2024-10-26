from __future__ import annotations

import random
from typing import Any

import cv2
import numpy as np

from albumentations.augmentations.mixing import functional as fmixing
from albumentations.core.bbox_utils import check_bboxes, denormalize_bboxes
from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.types import LENGTH_RAW_BBOX, Targets

__all__ = ["OverlayElements"]


class OverlayElements(DualTransform):
    """Apply overlay elements such as images and masks onto an input image. This transformation can be used to add
    various objects (e.g., stickers, logos) to images with optional masks and bounding boxes for better placement
    control.

    Args:
        metadata_key (str): Additional target key for metadata. Default `overlay_metadata`.
        p (float): Probability of applying the transformation. Default: 0.5.

    Possible Metadata Fields:
        - image (np.ndarray): The overlay image to be applied. This is a required field.
        - bbox (list[int]): The bounding box specifying the region where the overlay should be applied. It should
                            contain four floats: [y_min, x_min, y_max, x_max]. If `label_id` is provided, it should
                            be appended as the fifth element in the bbox. BBox should be in Albumentations format,
                            that is the same as normalized Pascal VOC format
                            [x_min / width, y_min / height, x_max / width, y_max / height]
        - mask (np.ndarray): An optional mask that defines the non-rectangular region of the overlay image. If not
                             provided, the entire overlay image is used.
        - mask_id (int): An optional identifier for the mask. If provided, the regions specified by the mask will
                         be labeled with this identifier in the output mask.

    Targets:
        image, mask

    Image types:
        uint8, float32

    Reference:
        https://github.com/danaaubakirova/doc-augmentation

    """

    _targets = (Targets.IMAGE, Targets.MASK)

    class InitSchema(BaseTransformInitSchema):
        metadata_key: str

    def __init__(
        self,
        metadata_key: str = "overlay_metadata",
        p: float = 0.5,
        always_apply: bool | None = None,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.metadata_key = metadata_key

    @property
    def targets_as_params(self) -> list[str]:
        return [self.metadata_key]

    @staticmethod
    def preprocess_metadata(
        metadata: dict[str, Any],
        img_shape: tuple[int, int],
        random_state: random.Random,
    ) -> dict[str, Any]:
        overlay_image = metadata["image"]
        overlay_height, overlay_width = overlay_image.shape[:2]
        image_height, image_width = img_shape[:2]

        if "bbox" in metadata:
            bbox = metadata["bbox"]
            bbox_np = np.array([bbox])
            check_bboxes(bbox_np)
            denormalized_bbox = denormalize_bboxes(bbox_np, img_shape[:2])[0]

            x_min, y_min, x_max, y_max = (int(x) for x in denormalized_bbox[:4])

            if "mask" in metadata:
                mask = metadata["mask"]
                mask = cv2.resize(mask, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_NEAREST)
            else:
                mask = np.ones((y_max - y_min, x_max - x_min), dtype=np.uint8)

            overlay_image = cv2.resize(overlay_image, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_AREA)
            offset = (y_min, x_min)

            if len(bbox) == LENGTH_RAW_BBOX and "bbox_id" in metadata:
                bbox = [x_min, y_min, x_max, y_max, metadata["bbox_id"]]
            else:
                bbox = (x_min, y_min, x_max, y_max, *bbox[4:])
        else:
            if image_height < overlay_height or image_width < overlay_width:
                overlay_image = cv2.resize(overlay_image, (image_width, image_height), interpolation=cv2.INTER_AREA)
                overlay_height, overlay_width = overlay_image.shape[:2]

            mask = metadata["mask"] if "mask" in metadata else np.ones_like(overlay_image, dtype=np.uint8)

            max_x_offset = image_width - overlay_width
            max_y_offset = image_height - overlay_height

            offset_x = random_state.randint(0, max_x_offset)
            offset_y = random_state.randint(0, max_y_offset)

            offset = (offset_y, offset_x)

            bbox = [
                offset_x,
                offset_y,
                offset_x + overlay_width,
                offset_y + overlay_height,
            ]

            if "bbox_id" in metadata:
                bbox = [*bbox, metadata["bbox_id"]]

        result = {
            "overlay_image": overlay_image,
            "overlay_mask": mask,
            "offset": offset,
            "bbox": bbox,
        }

        if "mask_id" in metadata:
            result["mask_id"] = metadata["mask_id"]

        return result

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        metadata = data[self.metadata_key]
        img_shape = params["shape"]

        if isinstance(metadata, list):
            overlay_data = [self.preprocess_metadata(md, img_shape, self.py_random) for md in metadata]
        else:
            overlay_data = [self.preprocess_metadata(metadata, img_shape, self.py_random)]

        return {
            "overlay_data": overlay_data,
        }

    def apply(
        self,
        img: np.ndarray,
        overlay_data: list[dict[str, Any]],
        **params: Any,
    ) -> np.ndarray:
        for data in overlay_data:
            overlay_image = data["overlay_image"]
            overlay_mask = data["overlay_mask"]
            offset = data["offset"]
            img = fmixing.copy_and_paste_blend(img, overlay_image, overlay_mask, offset=offset)
        return img

    def apply_to_mask(
        self,
        mask: np.ndarray,
        overlay_data: list[dict[str, Any]],
        **params: Any,
    ) -> np.ndarray:
        for data in overlay_data:
            if "mask_id" in data and data["mask_id"] is not None:
                overlay_mask = data["overlay_mask"]
                offset = data["offset"]
                mask_id = data["mask_id"]

                y_min, x_min = offset
                y_max = y_min + overlay_mask.shape[0]
                x_max = x_min + overlay_mask.shape[1]

                mask_section = mask[y_min:y_max, x_min:x_max]
                mask_section[overlay_mask > 0] = mask_id

        return mask

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("metadata_key",)
