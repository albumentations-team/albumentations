from __future__ import annotations

import random
import types
from typing import Any, Callable, Generator, Iterable, Iterator, Sequence
from warnings import warn

import cv2
import numpy as np
from albucore.functions import add_weighted
from albucore.utils import is_grayscale_image
from typing_extensions import Annotated

from albumentations.augmentations.mixing import functional as fmixing
from albumentations.core.bbox_utils import check_bbox, denormalize_bbox
from albumentations.core.transforms_interface import BaseTransformInitSchema, ReferenceBasedTransform
from albumentations.core.types import LENGTH_RAW_BBOX, BoxType, KeypointType, ReferenceImage, SizeType, Targets
from albumentations.random_utils import beta

from pydantic import Field

__all__ = ["MixUp", "OverlayElements"]


class MixUp(ReferenceBasedTransform):
    """Performs MixUp data augmentation, blending images, masks, and class labels with reference data.

    MixUp augmentation linearly combines an input (image, mask, and class label) with another set from a predefined
    reference dataset. The mixing degree is controlled by a parameter λ (lambda), sampled from a Beta distribution.
    This method is known for improving model generalization by promoting linear behavior between classes and
    smoothing decision boundaries.

    Reference:
        - Zhang, H., Cisse, M., Dauphin, Y.N., and Lopez-Paz, D. (2018). mixup: Beyond Empirical Risk Minimization.
        In International Conference on Learning Representations. https://arxiv.org/abs/1710.09412

    Args:
        reference_data (Optional[Union[Generator[ReferenceImage, None, None], Sequence[Any]]]):
            A sequence or generator of dictionaries containing the reference data for mixing
            If None or an empty sequence is provided, no operation is performed and a warning is issued.
        read_fn (Callable[[ReferenceImage], dict[str, Any]]):
            A function to process items from reference_data. It should accept items from reference_data
            and return a dictionary containing processed data:
                - The returned dictionary must include an 'image' key with a numpy array value.
                - It may also include 'mask', 'global_label' each associated with numpy array values.
            Defaults to a function that assumes input dictionary contains numpy arrays and directly returns it.
        mix_coef_return_name (str): Name used for the applied alpha coefficient in the returned dictionary.
            Defaults to "mix_coef".
        alpha (float):
            The alpha parameter for the Beta distribution, influencing the mix's balance. Must be ≥ 0.
            Higher values lead to more uniform mixing. Defaults to 0.4.
        p (float):
            The probability of applying the transformation. Defaults to 0.5.

    Targets:
        image, mask, global_label

    Image types:
        - uint8, float32

    Raises:
        - ValueError: If the alpha parameter is negative.
        - NotImplementedError: If the transform is applied to bounding boxes or keypoints.

    Notes:
        - If no reference data is provided, a warning is issued, and the transform acts as a no-op.
        - Notes if images are in float32 format, they should be within [0, 1] range.

    Example Usage:
        import albumentations as A
        import numpy as np
        from albumentations.core.types import ReferenceImage

        # Prepare reference data
        # Note: This code generates random reference data for demonstration purposes only.
        # In real-world applications, it's crucial to use meaningful and representative data.
        # The quality and relevance of your input data significantly impact the effectiveness
        # of the augmentation process. Ensure your data closely aligns with your specific
        # use case and application requirements.
        reference_data = [ReferenceImage(image=np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8),
                                         mask=np.random.randint(0, 4, (100, 100, 1), dtype=np.uint8),
                                         global_label=np.random.choice([0, 1], size=3)) for i in range(10)]

        # In this example, the lambda function simply returns its input, which works well for
        # data already in the expected format. For more complex scenarios, where the data might not be in
        # the required format or additional processing is needed, a more sophisticated function can be implemented.
        # Below is a hypothetical example where the input data is a file path, # and the function reads the image
        # file, converts it to a specific format, and possibly performs other preprocessing steps.

        # Example of a more complex read_fn that reads an image from a file path, converts it to RGB, and resizes it.
        # def custom_read_fn(file_path):
        #     from PIL import Image
        #     image = Image.open(file_path).convert('RGB')
        #     image = image.resize((100, 100))  # Example resize, adjust as needed.
        #     return np.array(image)

        # aug = A.Compose([A.RandomRotate90(), A.MixUp(p=1, reference_data=reference_data, read_fn=lambda x: x)])

        # For simplicity, the original lambda function is used in this example.
        # Replace `lambda x: x` with `custom_read_fn`if you need to process the data more extensively.

        # Apply augmentations
        image = np.empty([100, 100, 3], dtype=np.uint8)
        mask = np.empty([100, 100], dtype=np.uint8)
        global_label = np.array([0, 1, 0])
        data = aug(image=image, global_label=global_label, mask=mask)
        transformed_image = data["image"]
        transformed_mask = data["mask"]
        transformed_global_label = data["global_label"]

        # Print applied mix coefficient
        print(data["mix_coef"])  # Output: e.g., 0.9991580344142427
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.GLOBAL_LABEL)

    class InitSchema(BaseTransformInitSchema):
        reference_data: Generator[Any, None, None] | Sequence[Any] | None = None
        read_fn: Callable[[ReferenceImage], Any]
        alpha: Annotated[float, Field(default=0.4, ge=0, le=1)]
        mix_coef_return_name: str = "mix_coef"

    def __init__(
        self,
        reference_data: Generator[Any, None, None] | Sequence[Any] | None = None,
        read_fn: Callable[[ReferenceImage], Any] = lambda x: {"image": x, "mask": None, "class_label": None},
        alpha: float = 0.4,
        mix_coef_return_name: str = "mix_coef",
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.mix_coef_return_name = mix_coef_return_name

        self.read_fn = read_fn
        self.alpha = alpha

        if reference_data is None:
            warn("No reference data provided for MixUp. This transform will act as a no-op.", stacklevel=2)
            # Create an empty generator
            self.reference_data: list[Any] = []
        elif (
            isinstance(reference_data, types.GeneratorType)
            or isinstance(reference_data, Iterable)
            and not isinstance(reference_data, str)
        ):
            self.reference_data = reference_data  # type: ignore[assignment]
        else:
            msg = "reference_data must be a list, tuple, generator, or None."
            raise TypeError(msg)

    def apply(self, img: np.ndarray, mix_data: ReferenceImage, mix_coef: float, **params: Any) -> np.ndarray:
        if not mix_data:
            return img

        mix_img = mix_data["image"]

        if img.shape != mix_img.shape and not is_grayscale_image(img):
            msg = "The shape of the reference image should be the same as the input image."
            raise ValueError(msg)

        return add_weighted(img, mix_coef, mix_img.reshape(img.shape), 1 - mix_coef) if mix_img is not None else img

    def apply_to_mask(self, mask: np.ndarray, mix_data: ReferenceImage, mix_coef: float, **params: Any) -> np.ndarray:
        mix_mask = mix_data.get("mask")
        return (
            add_weighted(mask, mix_coef, mix_mask.reshape(mask.shape), 1 - mix_coef) if mix_mask is not None else mask
        )

    def apply_to_global_label(
        self,
        label: np.ndarray,
        mix_data: ReferenceImage,
        mix_coef: float,
        **params: Any,
    ) -> np.ndarray:
        mix_label = mix_data.get("global_label")
        if mix_label is not None and label is not None:
            return mix_coef * label + (1 - mix_coef) * mix_label
        return label

    def apply_to_bboxes(self, bboxes: Sequence[BoxType], mix_data: ReferenceImage, **params: Any) -> Sequence[BoxType]:
        msg = "MixUp does not support bounding boxes yet, feel free to submit pull request to https://github.com/albumentations-team/albumentations/."
        raise NotImplementedError(msg)

    def apply_to_keypoints(
        self,
        keypoints: Sequence[KeypointType],
        *args: Any,
        **params: Any,
    ) -> Sequence[KeypointType]:
        msg = "MixUp does not support keypoints yet, feel free to submit pull request to https://github.com/albumentations-team/albumentations/."
        raise NotImplementedError(msg)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "reference_data", "alpha"

    def get_params(self) -> dict[str, None | float | dict[str, Any]]:
        mix_data = None
        # Check if reference_data is not empty and is a sequence (list, tuple, np.array)
        if isinstance(self.reference_data, Sequence) and not isinstance(self.reference_data, (str, bytes)):
            if len(self.reference_data) > 0:  # Additional check to ensure it's not empty
                mix_idx = random.randint(0, len(self.reference_data) - 1)
                mix_data = self.reference_data[mix_idx]
        # Check if reference_data is an iterator or generator
        elif isinstance(self.reference_data, Iterator):
            try:
                mix_data = next(self.reference_data)  # Attempt to get the next item
            except StopIteration:
                warn(
                    "Reference data iterator/generator has been exhausted. "
                    "Further mixing augmentations will not be applied.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return {"mix_data": {}, "mix_coef": 1}

        # If mix_data is None or empty after the above checks, return default values
        if mix_data is None:
            return {"mix_data": {}, "mix_coef": 1}

        # If mix_data is not None, calculate mix_coef and apply read_fn
        mix_coef = beta(self.alpha, self.alpha)  # Assuming beta is defined elsewhere
        return {"mix_data": self.read_fn(mix_data), "mix_coef": mix_coef}

    def apply_with_params(self, params: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        res = super().apply_with_params(params, *args, **kwargs)
        if self.mix_coef_return_name:
            res[self.mix_coef_return_name] = params["mix_coef"]
        return res


class OverlayElements(ReferenceBasedTransform):
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
    def preprocess_metadata(metadata: dict[str, Any], img_shape: SizeType) -> dict[str, Any]:
        overlay_image = metadata["image"]
        overlay_height, overlay_width = overlay_image.shape[:2]
        image_height, image_width = img_shape[:2]

        if "bbox" in metadata:
            bbox = metadata["bbox"]
            check_bbox(bbox)
            denormalized_bbox = denormalize_bbox(bbox[:4], rows=image_height, cols=image_width)

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

            offset_x = random.randint(0, max_x_offset)
            offset_y = random.randint(0, max_y_offset)

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
            overlay_data = [self.preprocess_metadata(md, img_shape) for md in metadata]
        else:
            overlay_data = [self.preprocess_metadata(metadata, img_shape)]

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
