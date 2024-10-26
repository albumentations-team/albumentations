from __future__ import annotations

import math
from typing import Any, cast

import cv2
import numpy as np
from typing_extensions import Literal

from albumentations.augmentations.crops import functional as fcrops
from albumentations.augmentations.geometric.transforms import Affine
from albumentations.core.pydantic import BorderModeType, InterpolationType, SymmetricRangeType
from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.types import (
    ColorType,
    ScaleFloatType,
    Targets,
)

from . import functional as fgeometric

__all__ = ["Rotate", "RandomRotate90", "SafeRotate"]

SMALL_NUMBER = 1e-10


class RandomRotate90(DualTransform):
    """Randomly rotate the input by 90 degrees zero or more times.

    Args:
        p: probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    def apply(self, img: np.ndarray, factor: int, **params: Any) -> np.ndarray:
        return fgeometric.rot90(img, factor)

    def get_params(self) -> dict[str, int]:
        # Random int in the range [0, 3]
        return {"factor": self.py_random.randint(0, 3)}

    def apply_to_bboxes(self, bboxes: np.ndarray, factor: int, **params: Any) -> np.ndarray:
        return fgeometric.bboxes_rot90(bboxes, factor)

    def apply_to_keypoints(self, keypoints: np.ndarray, factor: int, **params: Any) -> np.ndarray:
        return fgeometric.keypoints_rot90(keypoints, factor, params["shape"])

    def get_transform_init_args_names(self) -> tuple[()]:
        return ()


class RotateInitSchema(BaseTransformInitSchema):
    limit: SymmetricRangeType

    interpolation: InterpolationType
    mask_interpolation: InterpolationType

    border_mode: BorderModeType

    value: ColorType | None
    mask_value: ColorType | None


class Rotate(DualTransform):
    """Rotate the input by an angle selected randomly from the uniform distribution.

    Args:
        limit (float | tuple[float, float]): Range from which a random angle is picked. If limit is a single float,
            an angle is picked from (-limit, limit). Default: (-90, 90)
        interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): Flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of ints, list of float): Padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float, list of ints, list of float): Padding value if border_mode is cv2.BORDER_CONSTANT
            applied for masks.
        rotate_method (str): Method to rotate bounding boxes. Should be 'largest_box' or 'ellipse'.
            Default: 'largest_box'
        crop_border (bool): Whether to crop border after rotation. If True, the output image size might differ
            from the input. Default: False
        mask_interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Note:
        - The rotation angle is randomly selected for each execution within the range specified by 'limit'.
        - When 'crop_border' is False, the output image will have the same size as the input, potentially
          introducing black triangles in the corners.
        - When 'crop_border' is True, the output image is cropped to remove black triangles, which may result
          in a smaller image.
        - Bounding boxes are rotated and may change size or shape.
        - Keypoints are rotated around the center of the image.

    Mathematical Details:
        1. An angle θ is randomly sampled from the range specified by 'limit'.
        2. The image is rotated around its center by θ degrees.
        3. The rotation matrix R is:
           R = [cos(θ)  -sin(θ)]
               [sin(θ)   cos(θ)]
        4. Each point (x, y) in the image is transformed to (x', y') by:
           [x']   [cos(θ)  -sin(θ)] [x - cx]   [cx]
           [y'] = [sin(θ)   cos(θ)] [y - cy] + [cy]
           where (cx, cy) is the center of the image.
        5. If 'crop_border' is True, the image is cropped to the largest rectangle that fits inside the rotated image.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.Rotate(limit=45, p=1.0)
        >>> result = transform(image=image)
        >>> rotated_image = result['image']
        # rotated_image will be the input image rotated by a random angle between -45 and 45 degrees
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(RotateInitSchema):
        rotate_method: Literal["largest_box", "ellipse"]
        crop_border: bool

    def __init__(
        self,
        limit: ScaleFloatType = (-90, 90),
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_REFLECT_101,
        value: ColorType | None = None,
        mask_value: ColorType | None = None,
        rotate_method: Literal["largest_box", "ellipse"] = "largest_box",
        crop_border: bool = False,
        mask_interpolation: int = cv2.INTER_NEAREST,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.limit = cast(tuple[float, float], limit)
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.rotate_method = rotate_method
        self.crop_border = crop_border

    def apply(
        self,
        img: np.ndarray,
        matrix: np.ndarray,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        **params: Any,
    ) -> np.ndarray:
        img_out = fgeometric.warp_affine(
            img,
            matrix,
            self.interpolation,
            self.value,
            self.border_mode,
            params["shape"][:2],
        )
        if self.crop_border:
            return fcrops.crop(img_out, x_min, y_min, x_max, y_max)
        return img_out

    def apply_to_mask(
        self,
        mask: np.ndarray,
        matrix: np.ndarray,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        **params: Any,
    ) -> np.ndarray:
        img_out = fgeometric.warp_affine(
            mask,
            matrix,
            self.mask_interpolation,
            self.mask_value,
            self.border_mode,
            params["shape"][:2],
        )
        if self.crop_border:
            return fcrops.crop(img_out, x_min, y_min, x_max, y_max)
        return img_out

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        bbox_matrix: np.ndarray,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        **params: Any,
    ) -> np.ndarray:
        image_shape = params["shape"][:2]
        bboxes_out = fgeometric.bboxes_affine(
            bboxes,
            bbox_matrix,
            self.rotate_method,
            image_shape,
            self.border_mode,
            image_shape,
        )
        if self.crop_border:
            return fcrops.crop_bboxes_by_coords(bboxes_out, (x_min, y_min, x_max, y_max), image_shape)
        return bboxes_out

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        matrix: np.ndarray,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        **params: Any,
    ) -> np.ndarray:
        keypoints_out = fgeometric.keypoints_affine(
            keypoints,
            matrix,
            params["shape"][:2],
            scale={"x": 1, "y": 1},
            mode=self.border_mode,
        )
        if self.crop_border:
            return fcrops.crop_keypoints_by_coords(keypoints_out, (x_min, y_min, x_max, y_max))
        return keypoints_out

    @staticmethod
    def _rotated_rect_with_max_area(height: int, width: int, angle: float) -> dict[str, int]:
        """Given a rectangle of size wxh that has been rotated by 'angle' (in
        degrees), computes the width and height of the largest possible
        axis-aligned rectangle (maximal area) within the rotated rectangle.

        Reference:
            https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
        """
        angle = math.radians(angle)
        width_is_longer = width >= height
        side_long, side_short = (width, height) if width_is_longer else (height, width)

        # since the solutions for angle, -angle and 180-angle are all the same,
        # it is sufficient to look at the first quadrant and the absolute values of sin,cos:
        sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
        if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < SMALL_NUMBER:
            # half constrained case: two crop corners touch the longer side,
            # the other two corners are on the mid-line parallel to the longer line
            x = 0.5 * side_short
            wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
        else:
            # fully constrained case: crop touches all 4 sides
            cos_2a = cos_a * cos_a - sin_a * sin_a
            wr, hr = (width * cos_a - height * sin_a) / cos_2a, (height * cos_a - width * sin_a) / cos_2a

        return {
            "x_min": max(0, int(width / 2 - wr / 2)),
            "x_max": min(width, int(width / 2 + wr / 2)),
            "y_min": max(0, int(height / 2 - hr / 2)),
            "y_max": min(height, int(height / 2 + hr / 2)),
        }

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        angle = self.py_random.uniform(*self.limit)

        if self.crop_border:
            height, width = params["shape"][:2]
            out_params = self._rotated_rect_with_max_area(height, width, angle)
        else:
            out_params = {"x_min": -1, "x_max": -1, "y_min": -1, "y_max": -1}

        center = fgeometric.center(params["shape"][:2])
        bbox_center = fgeometric.center_bbox(params["shape"][:2])

        translate: fgeometric.TranslateDict = {"x": 0, "y": 0}
        shear: fgeometric.ShearDict = {"x": 0, "y": 0}
        scale: fgeometric.ScaleDict = {"x": 1, "y": 1}
        rotate = angle

        matrix = fgeometric.create_affine_transformation_matrix(translate, shear, scale, rotate, center)
        bbox_matrix = fgeometric.create_affine_transformation_matrix(translate, shear, scale, rotate, bbox_center)
        out_params["matrix"] = matrix
        out_params["bbox_matrix"] = bbox_matrix

        return out_params

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "limit",
            "interpolation",
            "border_mode",
            "value",
            "mask_value",
            "rotate_method",
            "crop_border",
            "mask_interpolation",
        )


class SafeRotate(Affine):
    """Rotate the input inside the input's frame by an angle selected randomly from the uniform distribution.

    This transformation ensures that the entire rotated image fits within the original frame by scaling it
    down if necessary. The resulting image maintains its original dimensions but may contain artifacts due to the
    rotation and scaling process.

    Args:
        limit (float | tuple[float, float]): Range from which a random angle is picked. If limit is a single float,
            an angle is picked from (-limit, limit). Default: (-90, 90)
        interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): Flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of int, list of float): Padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float, list of int, list of float): Padding value if border_mode is cv2.BORDER_CONSTANT applied
            for masks.
        rotate_method (str): Method to rotate bounding boxes. Should be 'largest_box' or 'ellipse'.
            Default: 'largest_box'
        mask_interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Note:
        - The rotation is performed around the center of the image.
        - After rotation, the image is scaled to fit within the original frame, which may cause some distortion.
        - The output image will always have the same dimensions as the input image.
        - Bounding boxes and keypoints are transformed along with the image.

    Mathematical Details:
        1. An angle θ is randomly sampled from the range specified by 'limit'.
        2. The image is rotated around its center by θ degrees.
        3. The rotation matrix R is:
           R = [cos(θ)  -sin(θ)]
               [sin(θ)   cos(θ)]
        4. The scaling factor s is calculated to ensure the rotated image fits within the original frame:
           s = min(width / (width * |cos(θ)| + height * |sin(θ)|),
                   height / (width * |sin(θ)| + height * |cos(θ)|))
        5. The combined transformation matrix T is:
           T = [s*cos(θ)  -s*sin(θ)  tx]
               [s*sin(θ)   s*cos(θ)  ty]
           where tx and ty are translation factors to keep the image centered.
        6. Each point (x, y) in the image is transformed to (x', y') by:
           [x']   [s*cos(θ)   s*sin(θ)] [x - cx]   [cx]
           [y'] = [-s*sin(θ)  s*cos(θ)] [y - cy] + [cy]
           where (cx, cy) is the center of the image.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.SafeRotate(limit=45, p=1.0)
        >>> result = transform(image=image)
        >>> rotated_image = result['image']
        # rotated_image will be the input image rotated by a random angle between -45 and 45 degrees,
        # scaled to fit within the original 100x100 frame
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(RotateInitSchema):
        rotate_method: Literal["largest_box", "ellipse"]

    def __init__(
        self,
        limit: ScaleFloatType = (-90, 90),
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_REFLECT_101,
        value: ColorType | None = None,
        mask_value: ColorType | None = None,
        rotate_method: Literal["largest_box", "ellipse"] = "largest_box",
        mask_interpolation: int = cv2.INTER_NEAREST,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        value = 0 if value is None else value
        mask_value = 0 if mask_value is None else mask_value
        super().__init__(
            rotate=limit,
            interpolation=interpolation,
            mode=border_mode,
            cval=value,
            cval_mask=mask_value,
            rotate_method=rotate_method,
            fit_output=True,
            mask_interpolation=mask_interpolation,
            p=p,
            always_apply=always_apply,
        )
        self.limit = cast(tuple[float, float], limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.rotate_method = rotate_method
        self.mask_interpolation = mask_interpolation

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "limit",
            "interpolation",
            "border_mode",
            "value",
            "mask_value",
            "rotate_method",
            "mask_interpolation",
        )

    def _create_safe_rotate_matrix(
        self,
        angle: float,
        center: tuple[float, float],
        image_shape: tuple[int, int],
    ) -> tuple[np.ndarray, dict[str, float]]:
        height, width = image_shape[:2]
        rotation_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new image size
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])
        new_w = int(height * abs_sin + width * abs_cos)
        new_h = int(height * abs_cos + width * abs_sin)

        # Adjust the rotation matrix to take into account the new size
        rotation_mat[0, 2] += new_w / 2 - center[0]
        rotation_mat[1, 2] += new_h / 2 - center[1]

        # Calculate scaling factors
        scale_x = width / new_w
        scale_y = height / new_h

        # Create scaling matrix
        scale_mat = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])

        # Combine rotation and scaling
        matrix = scale_mat @ np.vstack([rotation_mat, [0, 0, 1]])

        return matrix, {"x": scale_x, "y": scale_y}

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        image_shape = params["shape"][:2]
        angle = self.py_random.uniform(*self.limit)

        # Calculate centers for image and bbox
        image_center = fgeometric.center(image_shape)
        bbox_center = fgeometric.center_bbox(image_shape)

        # Create matrices for image and bbox
        matrix, scale = self._create_safe_rotate_matrix(angle, image_center, image_shape)
        bbox_matrix, _ = self._create_safe_rotate_matrix(angle, bbox_center, image_shape)

        return {
            "rotate": angle,
            "scale": scale,
            "matrix": matrix,
            "bbox_matrix": bbox_matrix,
            "output_shape": image_shape,
        }
