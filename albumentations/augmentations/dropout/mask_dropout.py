import random
from typing import Any, Dict, Tuple, Union

import cv2
import numpy as np
from skimage.measure import label

from ...core.transforms_interface import DualTransform, to_tuple

__all__ = ["MaskDropout"]


class MaskDropout(DualTransform):
    """
    Image & mask augmentation that zero out mask and image regions corresponding
    to randomly chosen object instance from mask.

    Mask must be single-channel image, zero values treated as background.
    Image can be any number of channels.

    Inspired by https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/114254

    Args:
        max_objects: Maximum number of labels that can be zeroed out. Can be tuple, in this case it's [min, max]
        image_fill_value: Fill value to use when filling image.
            Can be 'inpaint' to apply inpaining (works only  for 3-chahnel images)
        mask_fill_value: Fill value to use when filling mask.

    Targets:
        image, mask

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        max_objects: int = 1,
        image_fill_value: Union[int, float, str] = 0,
        mask_fill_value: Union[int, float] = 0,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super(MaskDropout, self).__init__(always_apply, p)
        self.max_objects = to_tuple(max_objects, 1)
        self.image_fill_value = image_fill_value
        self.mask_fill_value = mask_fill_value

    @property
    def targets_as_params(self):
        return ["mask"]

    def get_params_dependent_on_targets(self, params) -> Dict[str, Any]:
        mask = params["mask"]

        label_image, num_labels = label(mask, return_num=True)

        if num_labels == 0:
            dropout_mask = None
        else:
            objects_to_drop = random.randint(int(self.max_objects[0]), int(self.max_objects[1]))
            objects_to_drop = min(num_labels, objects_to_drop)

            if objects_to_drop == num_labels:
                dropout_mask = mask > 0
            else:
                labels_index = random.sample(range(1, num_labels + 1), objects_to_drop)
                dropout_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=bool)
                for label_index in labels_index:
                    dropout_mask |= label_image == label_index

        params.update({"dropout_mask": dropout_mask})
        return params

    def apply(self, img: np.ndarray, dropout_mask: np.ndarray = None, **params) -> np.ndarray:
        if dropout_mask is None:
            return img

        if self.image_fill_value == "inpaint":
            dropout_mask = dropout_mask.astype(np.uint8)
            _, _, w, h = cv2.boundingRect(dropout_mask)
            radius = min(3, max(w, h) // 2)
            img = cv2.inpaint(img, dropout_mask, radius, cv2.INPAINT_NS)
        else:
            img = img.copy()
            img[dropout_mask] = self.image_fill_value

        return img

    def apply_to_mask(self, img: np.ndarray, dropout_mask: np.ndarray = None, **params) -> np.ndarray:
        if dropout_mask is None:
            return img

        img = img.copy()
        img[dropout_mask] = self.mask_fill_value
        return img

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return "max_objects", "image_fill_value", "mask_fill_value"
