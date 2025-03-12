from __future__ import annotations

import random
import types
from collections.abc import Generator, Iterable, Iterator, Sequence
from typing import Annotated, Any, Callable, Literal
from warnings import warn

import cv2
import numpy as np
from albucore import get_num_channels
from pydantic import AfterValidator

from albumentations.augmentations.mixing import functional as fmixing
from albumentations.core.bbox_utils import check_bboxes, denormalize_bboxes
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.type_definitions import LENGTH_RAW_BBOX, ReferenceImage, Targets

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
    ):
        super().__init__(p=p)
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


class Mosaic(DualTransform):
    """Combine four images into one in a mosaic style. This transformation requires a sequence of four images
    and their corresponding properties such as masks, bounding boxes, and keypoints.

    Args:
        #TODO

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Example:
        #TODO

    Reference:
        YOLOv4: Optimal Speed and Accuracy of Object Detection: https://arxiv.org/pdf/2004.10934
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        reference_data: Generator[Any, None, None] | Sequence[Any] | None = None
        read_fn: Callable[[ReferenceImage], Any]
        mosaic_size: int | tuple[int, int]
        center_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ]
        mask_interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ]
        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float

    def __init__(
        self,
        reference_data: Generator[Any, None, None] | Sequence[Any] | None = None,
        read_fn: Callable[[ReferenceImage], Any] = lambda x: x,
        mosaic_size: int | tuple[int, int] = 512,
        center_range: tuple[float, float] = (0.3, 0.7),
        keep_aspect_ratio: bool = False,
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ] = cv2.INTER_LINEAR,
        mask_interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ] = cv2.INTER_NEAREST,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.read_fn = read_fn
        self.center_range = center_range
        self.keep_aspect_ratio = keep_aspect_ratio
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.fill = fill
        self.fill_mask = fill_mask
        # Specifies how many images to use from reference_data to make the mosaic
        self.n = 3  # For now we focus on 2x2 mosaic

        if isinstance(mosaic_size, (list, tuple)):
            if len(mosaic_size) >= 2:
                self.mosaic_size = mosaic_size[:2]
            else:
                msg = "mosaic_size provided as a list or tuple must have at least two elements (height, width)."
                raise TypeError(msg)
        else:
            self.mosaic_size = (mosaic_size, mosaic_size)

        if reference_data is None:
            warn("No reference data provided for Mosaic. This transform will act as a no-op.", stacklevel=2)
            # Create an empty generator
            self.reference_data: Generator[Any, None, None] | Sequence[Any] = []
        elif isinstance(reference_data, types.GeneratorType) or (
            isinstance(reference_data, Iterable) and not isinstance(reference_data, (str, bytes))
        ):
            self.reference_data = reference_data
        else:
            msg = "reference_data must be a list, tuple, generator, or None."
            raise TypeError(msg)

    @property
    def targets_as_params(self) -> list[str]:
        return []

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ()

    def get_params(self) -> dict[str, tuple[int, int] | list[None | dict[str, Any]]]:
        mosaic_data = []

        # Check if reference_data is not empty and is a sequence (list, tuple, np.array)
        if isinstance(self.reference_data, Sequence) and not isinstance(self.reference_data, (str, bytes)):
            if len(self.reference_data) >= self.n:  # Additional check to ensure it's not empty and has enough elements
                mosaic_idxes = self.py_random.sample(range(len(self.reference_data)), self.n)
                mosaic_data = [self.reference_data[mosaic_idx] for mosaic_idx in mosaic_idxes]

        # Check if reference_data is an iterator or generator
        elif isinstance(self.reference_data, Iterator):
            try:
                for _ in range(self.n):
                    mosaic_data.append(next(self.reference_data))  # Attempt to get the next item
            except StopIteration:
                warn(
                    "Reference data iterator/generator has been exhausted. "
                    "Further mosaic augmentations will not be applied.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return {"mosaic_data": [], "center_pt": (0, 0)}

        # If mosaic_data has not enough data after the above checks, return default values
        if len(mosaic_data) < self.n:
            return {"mosaic_data": [], "center_pt": (0, 0)}

        # If mosaic_data is not empty, calculate center_pt and apply read_fn
        mosaic_h, mosaic_w = self.mosaic_size  # We force the center to be in the left half of the 2x2 mosaic
        center_x = int(mosaic_w * self.py_random.uniform(*self.center_range))
        center_y = int(mosaic_h * self.py_random.uniform(*self.center_range))
        return {"mosaic_data": [self.read_fn(md) for md in mosaic_data[: self.n]], "center_pt": (center_x, center_y)}

    def apply(
        self,
        img: np.ndarray,
        mosaic_data: list[ReferenceImage],
        center_point: tuple[int, int],
        **params: Any,
    ) -> np.ndarray:
        if len(mosaic_data) == 0:  # No-op
            return img

        add_images = [md["image"] for md in mosaic_data]
        for add_img in add_images:
            if get_num_channels(add_img) != get_num_channels(img):
                msg = "The number of channels of the mosaic image should be the same as the input image."
                raise ValueError(msg)

            if add_img.dtype != img.dtype:
                msg = "The data type of the mosaic image should be the same as the input image."
                raise ValueError(msg)

        four_imgs = [img]
        four_imgs.extend(add_images)

        return fmixing.create_2x2_mosaic_image(
            four_imgs,
            center_pt=center_point,
            mosaic_size=self.mosaic_size,
            keep_aspect_ratio=self.keep_aspect_ratio,
            interpolation=self.interpolation,
            fill=self.fill,
        )

    def apply_to_mask(
        self,
        mask: np.ndarray,
        mosaic_data: list[ReferenceImage],
        center_point: tuple[int, int],
        **params: Any,
    ) -> np.ndarray:
        if len(mosaic_data) == 0:  # No-op
            return mask

        add_masks = [md.get("mask") for md in mosaic_data]

        for add_mask in add_masks:
            if add_mask is None:
                return mask  # No-op

            if get_num_channels(add_mask) != get_num_channels(mask):
                msg = "The number of channels of the mosaic mask should be the same as the input mask."
                raise ValueError(msg)

            if add_mask.dtype != mask.dtype:
                msg = "The data type of the mosaic mask should be the same as the input mask."
                raise ValueError(msg)

        four_masks = [mask]
        four_masks.extend(add_masks)

        if self.fill_mask is None:  # Handles incompatible type Union[tuple[float, ...], float, None]
            self.fill_mask = 0.0

        return fmixing.create_2x2_mosaic_image(
            four_masks,
            center_pt=center_point,
            mosaic_size=self.mosaic_size,
            keep_aspect_ratio=self.keep_aspect_ratio,
            interpolation=self.mask_interpolation,
            fill=self.fill_mask,
        )

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        mosaic_data: list[ReferenceImage],
        center_point: tuple[int, int],
        **params: Any,
    ) -> np.ndarray:
        if len(mosaic_data) == 0:  # No-op
            return bboxes

        all_bboxes = [bboxes]
        for md in mosaic_data:
            add_bboxes = md.get("bbox")
            if add_bboxes is None:
                all_bboxes.append(None)  # Need to know which quadrant does not have bbox
                continue

            if isinstance(add_bboxes, tuple):
                add_bboxes = np.array([add_bboxes], dtype=np.float32)

            if add_bboxes.shape[1] != bboxes.shape[1]:
                msg = "The length of the mosaic bboxes should be the same as the input bboxes."
                raise ValueError(msg)

            all_bboxes.append(add_bboxes)

        all_img_shapes = [params["shape"][:2]]
        all_img_shapes.extend([md["image"].shape[:2] for md in mosaic_data])

        return fmixing.get_mosaic_bboxes(
            all_bboxes,
            all_img_shapes,
            center_pt=center_point,
            mosaic_size=self.mosaic_size,
            keep_aspect_ratio=self.keep_aspect_ratio,
        )

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        mosaic_data: list[ReferenceImage],
        center_point: tuple[int, int],
        **params: Any,
    ) -> np.ndarray:
        if len(mosaic_data) == 0:  # No-op
            return keypoints

        all_keypoints = [keypoints]
        for md in mosaic_data:
            add_keypoints = md.get("keypoints")
            if add_keypoints is None:
                all_keypoints.append(None)  # Need to know which quadrant does not have keypoints
                continue

            if isinstance(add_keypoints, tuple):
                add_keypoints = np.array([add_keypoints], dtype=np.float32)

            if add_keypoints.shape[1] != add_keypoints.shape[1]:
                msg = "The length of the mosaic keypoints should be the same as the input keypoints."
                raise ValueError(msg)

            all_keypoints.append(add_keypoints)

        all_img_shapes = [params["shape"][:2]]
        all_img_shapes.extend([md["image"].shape[:2] for md in mosaic_data])

        return fmixing.get_mosaic_keypoints(
            all_keypoints,
            all_img_shapes,
            center_pt=center_point,
            mosaic_size=self.mosaic_size,
        )
