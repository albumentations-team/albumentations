import math
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import skimage.transform
from scipy.ndimage import gaussian_filter

from albumentations.augmentations.utils import (
    _maybe_process_in_chunks,
    angles_2pi_range,
    clipped,
    preserve_channel_dim,
    preserve_shape,
)

from ... import random_utils
from ...core.bbox_utils import (
    denormalize_bboxes_np,
    ensure_bboxes_format,
    normalize_bboxes_np,
    use_bboxes_ndarray,
)
from ...core.keypoints_utils import ensure_keypoints_format, use_keypoints_ndarray
from ...core.transforms_interface import (
    BoxesArray,
    BoxType,
    FillValueType,
    ImageColorType,
    KeypointsArray,
    KeypointsInternalType,
)

__all__ = [
    "optical_distortion",
    "elastic_transform_approx",
    "grid_distortion",
    "pad",
    "pad_with_params",
    "keypoints_rot90",
    "rotate",
    "keypoints_rotate",
    "shift_scale_rotate",
    "keypoints_shift_scale_rotate",
    "bboxes_shift_scale_rotate",
    "elastic_transform",
    "resize",
    "scale",
    "py3round",
    "_func_max_size",
    "longest_max_size",
    "smallest_max_size",
    "perspective",
    "perspective_bboxes",
    "rotation2DMatrixToEulerAngles",
    "perspective_keypoints",
    "_is_identity_matrix",
    "warp_affine",
    "keypoints_affine",
    "bboxes_affine",
    "safe_rotate",
    "bboxes_safe_rotate",
    "keypoints_safe_rotate",
    "piecewise_affine",
    "to_distance_maps",
    "from_distance_maps",
    "keypoints_piecewise_affine",
    "bbox_piecewise_affine",
    "bboxes_piecewise_affine",
    "bboxes_flip",
    "bboxes_hflip",
    "bboxes_vflip",
    "bboxes_transpose",
    "vflip",
    "hflip",
    "vflip_cv2",
    "hflip_cv2",
    "transpose",
    "keypoints_flip",
    "keypoints_vflip",
    "keypoints_hflip",
    "keypoints_transpose",
]


@ensure_bboxes_format
@use_bboxes_ndarray(return_array=True)
def bboxes_rot90(bboxes: BoxesArray, factor: int, rows: int, cols: int) -> BoxesArray:
    if factor not in {0, 1, 2, 3}:
        raise ValueError("Parameter n must be in set {0, 1, 2, 3}")

    if not factor or not len(bboxes):
        return bboxes

    if factor == 1:
        bboxes[:, [0, 2]] = 1 - bboxes[:, [0, 2]]
        bboxes = bboxes[:, [1, 2, 3, 0]]
    elif factor == 2:
        bboxes = 1 - bboxes
        bboxes = bboxes[:, [2, 3, 0, 1]]
    elif factor == 3:
        bboxes[:, [1, 3]] = 1 - bboxes[:, [1, 3]]
        bboxes = bboxes[:, [3, 0, 1, 2]]
    return bboxes


@ensure_keypoints_format
@use_keypoints_ndarray(return_array=True)
@angles_2pi_range
def keypoints_rot90(keypoints: KeypointsArray, factor: int, rows: int, cols: int, **params) -> KeypointsArray:
    """Rotates a batch of keypoints by 90 degrees CCW (see np.rot90)

    Args:
        keypoints (KeypointsInternalType): A batch of keypoints in `(x, y, angle, scale)` format.
        factor (int): Number of CCW rotations. Must be in range [0;3] See np.rot90.
        rows (int): Image height.
        cols (int): Image width.

    Returns:
        KeypointsInternalType: A batch of keypoints in `(x, y, angle, scale)` format.

    Raises:
        ValueError: if factor not in set {0, 1, 2, 3}

    """
    if factor not in {0, 1, 2, 3}:
        raise ValueError("Parameter n must be in set {0, 1, 2, 3}")

    if factor == 1:
        keypoints[..., 2] -= math.pi / 2
        keypoints[..., 0] = cols - 1 - keypoints[..., 0]
        keypoints[..., [0, 1]] = keypoints[..., [1, 0]]
    elif factor == 2:
        keypoints[..., 2] -= math.pi
        keypoints[..., 0] = cols - 1 - keypoints[..., 0]
        keypoints[..., 1] = rows - 1 - keypoints[..., 1]
    elif factor == 3:
        keypoints[..., 2] += math.pi / 2
        keypoints[..., 1] = rows - 1 - keypoints[..., 1]
        keypoints[..., [0, 1]] = keypoints[..., [1, 0]]

    return keypoints


@preserve_channel_dim
def rotate(
    img: np.ndarray,
    angle: float,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_REFLECT_101,
    value: Optional[ImageColorType] = None,
):
    height, width = img.shape[:2]
    # for images we use additional shifts of (0.5, 0.5) as otherwise
    # we get an ugly black border for 90deg rotations
    matrix = cv2.getRotationMatrix2D((width / 2 - 0.5, height / 2 - 0.5), angle, 1.0)

    warp_fn = _maybe_process_in_chunks(
        cv2.warpAffine, M=matrix, dsize=(width, height), flags=interpolation, borderMode=border_mode, borderValue=value
    )
    return warp_fn(img)


@ensure_bboxes_format
@use_bboxes_ndarray(return_array=True)
def bboxes_rotate(bboxes: BoxesArray, angle: float, method: str, rows: int, cols: int) -> BoxesArray:
    """Rotates a batch of bounding boxes by angle degrees.

    Args:
        bboxes: A batch of bounding boxes in `albumentations` format.
        angle: Angle of rotation in degrees.
        method: Rotation method used. Should be one of: "largest_box", "ellipse". Default: "largest_box".
        rows: Image rows.
        cols: Image cols.

    Returns:
        A batch of bounding boxes in `albumentations` format.

    References:
        https://arxiv.org/abs/2109.13488

    """
    if not len(bboxes):
        return bboxes

    scale_ = cols / float(rows)
    if method == "largest_box":
        x = np.tile(bboxes[:, [0, 2]], 2) - 0.5
        y = np.tile(bboxes[:, [1, 3]], 2) - 0.5
    elif method == "ellipse":
        w = np.expand_dims((bboxes[:, 2] - bboxes[:, 0]) / 2.0, axis=0).transpose()
        h = np.expand_dims((bboxes[:, 3] - bboxes[:, 1]) / 2.0, axis=0).transpose()
        data = np.arange(0, 360, dtype=np.float32)
        x = w * np.sin(np.radians(data)) + (w + np.expand_dims(bboxes[:, 0], 0).transpose() - 0.5)
        y = h * np.cos(np.radians(data)) + (h + np.expand_dims(bboxes[:, 1], 0).transpose() - 0.5)
    else:
        raise ValueError(
            f"Method {method} is not a valid rotation method. Should only support `largest_box` or `ellipse`."
        )
    angle = np.deg2rad(angle)
    x_t = (x * np.cos(angle) * scale_ + y * np.sin(angle)) / scale_ + 0.5
    y_t = (x * -np.sin(angle) * scale_ + y * np.cos(angle)) + 0.5

    bboxes = np.concatenate(
        [
            [np.min(x_t, axis=1)],
            [np.max(x_t, axis=1)],
            [np.min(y_t, axis=1)],
            [np.max(y_t, axis=1)],
        ],
        axis=0,
    ).transpose()

    return bboxes


@ensure_keypoints_format
@use_keypoints_ndarray(return_array=True)
@angles_2pi_range
def keypoints_rotate(keypoints: KeypointsArray, angle: float, rows: int, cols: int, **params) -> KeypointsArray:
    """Rotate a keypoint by angle.

    Args:
        keypoints (KeypointsArray): A batch of keypoints in `(x, y, angle, scale)` format.
        angle (float): Rotation angle.
        rows (int): Image height.
        cols (int): Image width.

    Returns:
        KeypointsArray: A batch of keypoints in `(x, y, angle, scale)` format.

    """
    center = (cols - 1) * 0.5, (rows - 1) * 0.5
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    keypoints[..., [0, 1]] = cv2.transform(np.expand_dims(keypoints[..., [0, 1]], axis=0), matrix).squeeze()
    keypoints[..., 2] += math.radians(angle)
    return keypoints


@preserve_channel_dim
def shift_scale_rotate(
    img, angle, scale, dx, dy, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None
):
    height, width = img.shape[:2]
    # for images we use additional shifts of (0.5, 0.5) as otherwise
    # we get an ugly black border for 90deg rotations
    center = (width / 2 - 0.5, height / 2 - 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += dx * width
    matrix[1, 2] += dy * height

    warp_affine_fn = _maybe_process_in_chunks(
        cv2.warpAffine, M=matrix, dsize=(width, height), flags=interpolation, borderMode=border_mode, borderValue=value
    )
    return warp_affine_fn(img)


@ensure_keypoints_format
@use_keypoints_ndarray(return_array=True)
@angles_2pi_range
def keypoints_shift_scale_rotate(
    keypoints: KeypointsArray, angle: int, scale: float, dx: int, dy: int, rows: int, cols: int, **params
) -> KeypointsArray:
    if not len(keypoints):
        return keypoints

    center = (cols - 1) * 0.5, (rows - 1) * 0.5
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += dx * cols
    matrix[1, 2] += dy * rows

    keypoints[..., [0, 1]] = cv2.transform(np.expand_dims(keypoints[..., [0, 1]], axis=0), matrix).squeeze()
    keypoints[..., 2] += math.radians(angle)
    keypoints[..., 3] *= scale

    return keypoints


@ensure_bboxes_format
@use_bboxes_ndarray(return_array=True)
def bboxes_shift_scale_rotate(
    bboxes: BoxesArray, angle: int, scale_: int, dx: int, dy: int, rotate_method: str, rows: int, cols: int, **kwargs
) -> BoxesArray:
    if not len(bboxes):
        return bboxes
    center = (cols / 2, rows / 2)
    if rotate_method == "ellipse":
        bboxes = bboxes_rotate(bboxes, angle, rotate_method, rows, cols)
        matrix = cv2.getRotationMatrix2D(center, 0, scale_)
    elif rotate_method == "largest_box":
        matrix = cv2.getRotationMatrix2D(center, angle, scale_)
    else:
        raise ValueError(
            f"Method {rotate_method} is not a valid rotation method. Should only support `largest_box` or `ellipse`."
        )
    matrix[0, 2] += dx * cols
    matrix[1, 2] += dy * rows

    xs = bboxes[..., [0, 2, 2, 0]].T
    ys = bboxes[..., [1, 1, 3, 3]].T
    ones = np.ones_like(xs)

    points_ones = np.stack([xs, ys, ones], axis=0).transpose()
    points_ones[..., 0] *= cols
    points_ones[..., 1] *= rows
    tr_points = matrix.dot(points_ones.transpose((0, 2, 1))).transpose((1, 2, 0))
    tr_points[..., 0] /= cols
    tr_points[..., 1] /= rows

    bboxes = np.concatenate(
        [
            np.min(tr_points[..., [0, 1]], axis=1),
            np.max(tr_points[..., [0, 1]], axis=1),
        ],
        axis=-1,
    )

    return bboxes


@preserve_shape
def elastic_transform(
    img: np.ndarray,
    alpha: float,
    sigma: float,
    alpha_affine: float,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_REFLECT_101,
    value: Optional[ImageColorType] = None,
    random_state: Optional[np.random.RandomState] = None,
    approximate: bool = False,
    same_dxdy: bool = False,
):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    Based on https://gist.github.com/ernestum/601cdf56d2b424757de5

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
    """
    height, width = img.shape[:2]

    # Random affine
    center_square = np.array((height, width), dtype=np.float32) // 2
    square_size = min((height, width)) // 3
    alpha = float(alpha)
    sigma = float(sigma)
    alpha_affine = float(alpha_affine)

    pts1 = np.array(
        [
            center_square + square_size,
            [center_square[0] + square_size, center_square[1] - square_size],
            center_square - square_size,
        ],
        dtype=np.float32,
    )
    pts2 = pts1 + random_utils.uniform(-alpha_affine, alpha_affine, size=pts1.shape, random_state=random_state).astype(
        np.float32
    )
    matrix = cv2.getAffineTransform(pts1, pts2)

    warp_fn = _maybe_process_in_chunks(
        cv2.warpAffine, M=matrix, dsize=(width, height), flags=interpolation, borderMode=border_mode, borderValue=value
    )
    img = warp_fn(img)

    if approximate:
        # Approximate computation smooth displacement map with a large enough kernel.
        # On large images (512+) this is approximately 2X times faster
        dx = random_utils.rand(height, width, random_state=random_state).astype(np.float32) * 2 - 1
        cv2.GaussianBlur(dx, (17, 17), sigma, dst=dx)
        dx *= alpha
        if same_dxdy:
            # Speed up even more
            dy = dx
        else:
            dy = random_utils.rand(height, width, random_state=random_state).astype(np.float32) * 2 - 1
            cv2.GaussianBlur(dy, (17, 17), sigma, dst=dy)
            dy *= alpha
    else:
        dx = np.float32(
            gaussian_filter((random_utils.rand(height, width, random_state=random_state) * 2 - 1), sigma) * alpha
        )
        if same_dxdy:
            # Speed up
            dy = dx
        else:
            dy = np.float32(
                gaussian_filter((random_utils.rand(height, width, random_state=random_state) * 2 - 1), sigma) * alpha
            )

    x, y = np.meshgrid(np.arange(width), np.arange(height))

    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)

    remap_fn = _maybe_process_in_chunks(
        cv2.remap, map1=map_x, map2=map_y, interpolation=interpolation, borderMode=border_mode, borderValue=value
    )
    return remap_fn(img)


@preserve_channel_dim
def resize(img, height, width, interpolation=cv2.INTER_LINEAR):
    img_height, img_width = img.shape[:2]
    if height == img_height and width == img_width:
        return img
    resize_fn = _maybe_process_in_chunks(cv2.resize, dsize=(width, height), interpolation=interpolation)
    return resize_fn(img)


@preserve_channel_dim
def scale(img: np.ndarray, scale: float, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    height, width = img.shape[:2]
    new_height, new_width = int(height * scale), int(width * scale)
    return resize(img, new_height, new_width, interpolation)


@use_keypoints_ndarray(return_array=True)
def keypoints_scale(keypoints: KeypointsArray, scale_x: float, scale_y: float) -> KeypointsArray:
    """Scales a batch of keypoints by scale_x and scale_y.

    Args:
        keypoints (KeypointsArray): A batch of keypoints in `(x, y, angle, scale)` format.
        scale_x: Scale coefficient x-axis.
        scale_y: Scale coefficient y-axis.

    Returns:
        KeypointsArray, A batch of keypoints in `(x, y, angle, scale)` format.

    """
    if not len(keypoints):
        return keypoints
    scales = np.array([scale_x, scale_y, 1.0, max(scale_x, scale_y)])
    return keypoints * scales


def py3round(number):
    """Unified rounding in all python versions."""
    if abs(round(number) - number) == 0.5:
        return int(2.0 * round(number / 2.0))

    return int(round(number))


def _func_max_size(img, max_size, interpolation, func):
    height, width = img.shape[:2]

    scale = max_size / float(func(width, height))

    if scale != 1.0:
        new_height, new_width = tuple(py3round(dim * scale) for dim in (height, width))
        img = resize(img, height=new_height, width=new_width, interpolation=interpolation)
    return img


@preserve_channel_dim
def longest_max_size(img: np.ndarray, max_size: int, interpolation: int) -> np.ndarray:
    return _func_max_size(img, max_size, interpolation, max)


@preserve_channel_dim
def smallest_max_size(img: np.ndarray, max_size: int, interpolation: int) -> np.ndarray:
    return _func_max_size(img, max_size, interpolation, min)


@preserve_channel_dim
def perspective(
    img: np.ndarray,
    matrix: np.ndarray,
    max_width: int,
    max_height: int,
    border_val: Union[int, float, List[int], List[float], np.ndarray],
    border_mode: int,
    keep_size: bool,
    interpolation: int,
):
    h, w = img.shape[:2]
    perspective_func = _maybe_process_in_chunks(
        cv2.warpPerspective,
        M=matrix,
        dsize=(max_width, max_height),
        borderMode=border_mode,
        borderValue=border_val,
        flags=interpolation,
    )
    warped = perspective_func(img)

    if keep_size:
        return resize(warped, h, w, interpolation=interpolation)

    return warped


@ensure_bboxes_format
@use_bboxes_ndarray(return_array=True)
def perspective_bboxes(
    bboxes: BoxesArray,
    height: int,
    width: int,
    matrix: np.ndarray,
    max_width: int,
    max_height: int,
    keep_size: bool,
) -> BoxesArray:
    if not len(bboxes):
        return bboxes
    bboxes = denormalize_bboxes_np(bboxes, height, width)

    points = np.stack(
        [
            np.pad(bboxes[..., [0, 1]], ((0, 0), (0, 2))),
            np.pad(bboxes[..., [2, 1]], ((0, 0), (0, 2))),
            np.pad(bboxes[..., [2, 3]], ((0, 0), (0, 2))),
            np.pad(bboxes[..., [0, 3]], ((0, 0), (0, 2))),
        ],
        axis=1,
    )

    points = perspective_keypoints(
        points.reshape((-1, 4)),  # type: ignore[attr-defined]
        height=height,
        width=width,
        matrix=matrix,
        max_width=max_width,
        max_height=max_height,
        keep_size=keep_size,
    ).reshape(points.shape)

    bboxes = np.concatenate(
        [
            np.min(points[..., [0, 1]], axis=1),
            np.max(points[..., [0, 1]], axis=1),
        ],
        axis=-1,
    )

    return normalize_bboxes_np(bboxes, height if keep_size else max_height, width if keep_size else max_width)


def rotation2DMatrixToEulerAngles(matrix: np.ndarray, y_up: bool = False) -> float:
    """
    Args:
        matrix (np.ndarray): Rotation matrix
        y_up (bool): is Y axis looks up or down
    """
    if y_up:
        return np.arctan2(matrix[1, 0], matrix[0, 0])
    return np.arctan2(-matrix[1, 0], matrix[0, 0])


@use_keypoints_ndarray(return_array=True)
@angles_2pi_range
def perspective_keypoints(
    keypoints: KeypointsArray,
    height: int,
    width: int,
    matrix: np.ndarray,
    max_width: int,
    max_height: int,
    keep_size: bool,
) -> KeypointsArray:
    if not len(keypoints):
        return keypoints

    keypoints[..., [0, 1]] = cv2.perspectiveTransform(
        keypoints[..., [0, 1]].reshape(-1, 2)[np.newaxis, ...], matrix
    ).reshape(keypoints[..., [0, 1]].shape)
    keypoints[..., 2] += rotation2DMatrixToEulerAngles(matrix[:2, :2], y_up=True)

    scale_x = np.sign(matrix[0, 0]) * np.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2)
    scale_y = np.sign(matrix[1, 1]) * np.sqrt(matrix[1, 0] ** 2 + matrix[1, 1] ** 2)
    keypoints[..., 3] *= max(scale_x, scale_y)

    if keep_size:
        scale_x = width / max_width
        scale_y = height / max_height

        return keypoints_scale(keypoints, scale_x, scale_y)

    return keypoints


def _is_identity_matrix(matrix: skimage.transform.ProjectiveTransform) -> bool:
    return np.allclose(matrix.params, np.eye(3, dtype=np.float32))


@preserve_channel_dim
def warp_affine(
    image: np.ndarray,
    matrix: skimage.transform.ProjectiveTransform,
    interpolation: int,
    cval: Union[int, float, Sequence[int], Sequence[float]],
    mode: int,
    output_shape: Sequence[int],
) -> np.ndarray:
    if _is_identity_matrix(matrix):
        return image

    dsize = int(np.round(output_shape[1])), int(np.round(output_shape[0]))
    warp_fn = _maybe_process_in_chunks(
        cv2.warpAffine, M=matrix.params[:2], dsize=dsize, flags=interpolation, borderMode=mode, borderValue=cval
    )
    tmp = warp_fn(image)
    return tmp


# @angle_2pi_range
# def keypoint_affine(
#     keypoint: KeypointInternalType,
#     matrix: skimage.transform.ProjectiveTransform,
#     scale: dict,
# ) -> KeypointInternalType:
#     if _is_identity_matrix(matrix):
#         return keypoint
#
#     x, y, a, s = keypoint[:4]
#     x, y = cv2.transform(np.array([[[x, y]]]), matrix.params[:2]).squeeze()
#     a += rotation2DMatrixToEulerAngles(matrix.params[:2])
#     s *= np.max([scale["x"], scale["y"]])
#     return x, y, a, s


@ensure_keypoints_format
@use_keypoints_ndarray(return_array=True)
@angles_2pi_range
def keypoints_affine(
    keypoints: KeypointsArray,
    matrix: skimage.transform.ProjectiveTransform,
    scale: dict,
) -> KeypointsArray:
    if _is_identity_matrix(matrix):
        return keypoints

    keypoints[..., [0, 1]] = cv2.transform(np.expand_dims(keypoints[..., [0, 1]], axis=0), matrix.params[:2]).squeeze()
    keypoints[..., 2] += rotation2DMatrixToEulerAngles(matrix.params[:2])
    keypoints[..., 3] *= np.max([scale["x"], scale["y"]])
    return keypoints


@ensure_bboxes_format
@use_bboxes_ndarray(return_array=True)
def bboxes_affine(
    bboxes: BoxesArray,
    matrix: skimage.transform.ProjectiveTransform,
    rotate_method: str,
    rows: int,
    cols: int,
    output_shape: Sequence[int],
) -> BoxesArray:
    if _is_identity_matrix(matrix):
        return bboxes

    if not len(bboxes):
        return bboxes

    bboxes = denormalize_bboxes_np(bboxes, rows, cols)
    if rotate_method == "largest_box":
        points = np.stack(
            [
                bboxes[..., [0, 1]],
                bboxes[..., [2, 1]],
                bboxes[..., [2, 3]],
                bboxes[..., [0, 3]],
            ],
            axis=1,
        )  # points.shape == N * 4 * 2
    elif rotate_method == "ellipse":
        w = (bboxes[..., 2] - bboxes[..., 0]) / 2.0
        h = (bboxes[..., 3] - bboxes[..., 1]) / 2.0
        data = np.arange(0, 360, dtype=np.float32)
        x = w[np.newaxis, ...].T * np.sin(np.radians(data)) + (w + bboxes[..., 0] - 0.5).reshape((-1, 1))
        y = h[np.newaxis, ...].T * np.cos(np.radians(data)) + (h + bboxes[..., 1] - 0.5).reshape((-1, 1))
        points = np.stack([x, y], axis=2)
    else:
        raise ValueError(f"Method {rotate_method} is not a valid rotation method.")

    points = skimage.transform.matrix_transform(points.reshape(-1, 2), matrix.params).reshape(points.shape)

    bboxes = np.concatenate(
        [
            np.min(points[..., [0, 1]], axis=-2),
            np.max(points[..., [0, 1]], axis=-2),
        ],
        axis=-1,
    )

    return normalize_bboxes_np(bboxes, output_shape[0], output_shape[1])


@preserve_channel_dim
def safe_rotate(
    img: np.ndarray,
    matrix: np.ndarray,
    interpolation: int,
    value: FillValueType = None,
    border_mode: int = cv2.BORDER_REFLECT_101,
) -> np.ndarray:
    h, w = img.shape[:2]
    warp_fn = _maybe_process_in_chunks(
        cv2.warpAffine,
        M=matrix,
        dsize=(w, h),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=value,
    )
    return warp_fn(img)


@ensure_bboxes_format
@use_bboxes_ndarray(return_array=True)
def bboxes_safe_rotate(
    bboxes: BoxesArray,
    matrix: np.ndarray,
    rows: int,
    cols: int,
) -> BoxesArray:
    if not len(bboxes):
        return bboxes
    bboxes = denormalize_bboxes_np(bboxes, rows=rows, cols=cols)
    points = (
        np.stack(
            [
                np.stack([bboxes[..., 0], bboxes[..., 1], np.ones_like(bboxes[..., 1])], axis=1),
                np.stack([bboxes[..., 2], bboxes[..., 1], np.ones_like(bboxes[..., 1])], axis=1),
                np.stack([bboxes[..., 2], bboxes[..., 3], np.ones_like(bboxes[..., 1])], axis=1),
                np.stack([bboxes[..., 0], bboxes[..., 3], np.ones_like(bboxes[..., 1])], axis=1),
            ],
            axis=1,
        )
        @ matrix.T
    )

    bboxes = np.concatenate(
        [
            np.min(points[..., [0, 1]], axis=-2),
            np.max(points[..., [0, 1]], axis=-2),
        ],
        axis=-1,
    )

    return normalize_bboxes_np(bboxes, rows=rows, cols=cols)


@ensure_keypoints_format
@use_keypoints_ndarray(return_array=True)
def keypoints_safe_rotate(
    keypoints: KeypointsArray,
    matrix: np.ndarray,
    angle: float,
    scale_x: float,
    scale_y: float,
    cols: int,
    rows: int,
) -> KeypointsArray:
    if not len(keypoints):
        return keypoints
    keypoints[..., [0, 1]] = cv2.transform(np.expand_dims(keypoints[..., [0, 1]], axis=0), matrix).squeeze()
    keypoints[..., [0, 1]] = np.clip(keypoints[..., [0, 1]], [0, 0], [cols - 1, rows - 1])
    keypoints[..., 2] += angle
    keypoints[..., 3] *= max(scale_x, scale_y)
    return keypoints


@clipped
def piecewise_affine(
    img: np.ndarray,
    matrix: skimage.transform.PiecewiseAffineTransform,
    interpolation: int,
    mode: str,
    cval: float,
) -> np.ndarray:
    return skimage.transform.warp(
        img, matrix, order=interpolation, mode=mode, cval=cval, preserve_range=True, output_shape=img.shape
    )


def to_distance_maps(
    keypoints: Union[Sequence[Tuple[float, float]], np.ndarray], height: int, width: int, inverted: bool = False
) -> np.ndarray:
    """Generate a ``(H,W,N)`` array of distance maps for ``N`` keypoints.

    The ``n``-th distance map contains at every location ``(y, x)`` the
    euclidean distance to the ``n``-th keypoint.

    This function can be used as a helper when augmenting keypoints with a
    method that only supports the augmentation of images.

    Args:
        keypoints: keypoint coordinates
        height: image height
        width: image width
        inverted (bool): If ``True``, inverted distance maps are returned where each
            distance value d is replaced by ``d/(d+1)``, i.e. the distance
            maps have values in the range ``(0.0, 1.0]`` with ``1.0`` denoting
            exactly the position of the respective keypoint.

    Returns:
        (H, W, N) ndarray
            A ``float32`` array containing ``N`` distance maps for ``N``
            keypoints. Each location ``(y, x, n)`` in the array denotes the
            euclidean distance at ``(y, x)`` to the ``n``-th keypoint.
            If `inverted` is ``True``, the distance ``d`` is replaced
            by ``d/(d+1)``. The height and width of the array match the
            height and width in ``KeypointsOnImage.shape``.
    """
    distance_maps = np.zeros((height, width, len(keypoints)), dtype=np.float32)

    yy = np.arange(0, height)
    xx = np.arange(0, width)
    grid_xx, grid_yy = np.meshgrid(xx, yy)

    for i, (x, y) in enumerate(keypoints):
        distance_maps[:, :, i] = (grid_xx - x) ** 2 + (grid_yy - y) ** 2

    distance_maps = np.sqrt(distance_maps)
    if inverted:
        return 1 / (distance_maps + 1)
    return distance_maps


def from_distance_maps(
    distance_maps: np.ndarray,
    inverted: bool,
    if_not_found_coords: Optional[Union[Sequence[int], dict]],
    threshold: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Convert outputs of ``to_distance_maps()`` to ``KeypointsOnImage``.
    This is the inverse of `to_distance_maps`.

    Args:
        distance_maps (np.ndarray): The distance maps. ``N`` is the number of keypoints.
        inverted (bool): Whether the given distance maps were generated in inverted mode
            (i.e. :func:`KeypointsOnImage.to_distance_maps` was called with ``inverted=True``) or in non-inverted mode.
        if_not_found_coords (tuple, list, dict or None, optional):
            Coordinates to use for keypoints that cannot be found in `distance_maps`.

            * If this is a ``list``/``tuple``, it must contain two ``int`` values.
            * If it is a ``dict``, it must contain the keys ``x`` and ``y`` with each containing one ``int`` value.
            * If this is ``None``, then the keypoint will not be added.
        threshold (float): The search for keypoints works by searching for the
            argmin (non-inverted) or argmax (inverted) in each channel. This
            parameters contains the maximum (non-inverted) or minimum (inverted) value to accept in order to view a hit
            as a keypoint. Use ``None`` to use no min/max.
        nb_channels (None, int): Number of channels of the image on which the keypoints are placed.
            Some keypoint augmenters require that information. If set to ``None``, the keypoint's shape will be set
            to ``(height, width)``, otherwise ``(height, width, nb_channels)``.
    """
    if distance_maps.ndim != 3:
        raise ValueError(
            f"Expected three-dimensional input, "
            f"got {distance_maps.ndim} dimensions and shape {distance_maps.shape}."
        )
    height, width, nb_keypoints = distance_maps.shape

    drop_if_not_found = False
    if if_not_found_coords is None:
        drop_if_not_found = True
        if_not_found_x = -1
        if_not_found_y = -1
    elif isinstance(if_not_found_coords, (tuple, list)):
        if len(if_not_found_coords) != 2:
            raise ValueError(
                f"Expected tuple/list 'if_not_found_coords' to contain exactly two entries, "
                f"got {len(if_not_found_coords)}."
            )
        if_not_found_x = if_not_found_coords[0]
        if_not_found_y = if_not_found_coords[1]
    elif isinstance(if_not_found_coords, dict):
        if_not_found_x = if_not_found_coords["x"]
        if_not_found_y = if_not_found_coords["y"]
    else:
        raise ValueError(
            f"Expected if_not_found_coords to be None or tuple or list or dict, got {type(if_not_found_coords)}."
        )

    keypoints = []
    for i in range(nb_keypoints):
        if inverted:
            hitidx_flat = np.argmax(distance_maps[..., i])
        else:
            hitidx_flat = np.argmin(distance_maps[..., i])
        hitidx_ndim = np.unravel_index(hitidx_flat, (height, width))
        if not inverted and threshold is not None:
            found = distance_maps[hitidx_ndim[0], hitidx_ndim[1], i] < threshold
        elif inverted and threshold is not None:
            found = distance_maps[hitidx_ndim[0], hitidx_ndim[1], i] >= threshold
        else:
            found = True
        if found:
            keypoints.append((float(hitidx_ndim[1]), float(hitidx_ndim[0])))
        else:
            if not drop_if_not_found:
                keypoints.append((if_not_found_x, if_not_found_y))

    return keypoints


@ensure_keypoints_format
@use_keypoints_ndarray(return_array=True)
def keypoints_piecewise_affine(
    keypoints: KeypointsArray,
    matrix: skimage.transform.PiecewiseAffineTransform,
    h: int,
    w: int,
    keypoints_threshold: float,
) -> KeypointsArray:
    if not len(keypoints):
        return keypoints
    dist_maps = to_distance_maps(keypoints[..., [0, 1]], h, w, True)
    dist_maps = piecewise_affine(dist_maps, matrix, 0, "constant", 0)
    keypoints[..., [0, 1]] = np.array(from_distance_maps(dist_maps, True, {"x": -1, "y": -1}, keypoints_threshold))
    return keypoints


def bbox_piecewise_affine(
    bbox: BoxType,
    matrix: skimage.transform.PiecewiseAffineTransform,
    h: int,
    w: int,
    keypoints_threshold: float,
) -> BoxType:
    x1, y1, x2, y2 = denormalize_bboxes_np(np.array([bbox]), h, w)[0][:4]
    keypoints = [
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2),
    ]
    dist_maps = to_distance_maps(keypoints, h, w, True)
    dist_maps = piecewise_affine(dist_maps, matrix, 0, "constant", 0)
    keypoints = from_distance_maps(dist_maps, True, {"x": -1, "y": -1}, keypoints_threshold)
    keypoints = [i for i in keypoints if 0 <= i[0] < w and 0 <= i[1] < h]
    keypoints_arr = np.array(keypoints)
    x1 = keypoints_arr[:, 0].min()
    y1 = keypoints_arr[:, 1].min()
    x2 = keypoints_arr[:, 0].max()
    y2 = keypoints_arr[:, 1].max()
    return normalize_bboxes_np(np.array([[x1, y1, x2, y2]]), h, w)[0]


@ensure_bboxes_format
@use_bboxes_ndarray(return_array=True)
def bboxes_piecewise_affine(
    bboxes: BoxesArray,
    matrix: skimage.transform.PiecewiseAffineTransform,
    h: int,
    w: int,
    keypoints_threshold: float,
) -> BoxesArray:
    if not len(bboxes):
        return bboxes
    bboxes = denormalize_bboxes_np(bboxes, h, w)
    points = np.stack(
        [
            bboxes[..., [0, 1]],
            bboxes[..., [2, 1]],
            bboxes[..., [2, 3]],
            bboxes[..., [0, 3]],
        ],
        axis=1,
    ).reshape(
        (-1, 2)
    )  # points.shape == (N * 4) * 2
    dist_maps = to_distance_maps(points, h, w, True)
    dist_maps = piecewise_affine(dist_maps, matrix, 0, "constant", 0)
    points = from_distance_maps(dist_maps, True, {"x": -1, "y": -1}, keypoints_threshold)

    points = np.reshape(points, (-1, 4, 2))  # N * 4 * 2

    bboxes = np.concatenate(
        [
            np.min(points, axis=1),
            np.max(points, axis=1),
        ],
        axis=-1,
    )
    return normalize_bboxes_np(bboxes, h, w)


def vflip(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[::-1, ...])


def hflip(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[:, ::-1, ...])


def vflip_cv2(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 0)


def hflip_cv2(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 1)


@preserve_shape
def random_flip(img: np.ndarray, code: int) -> np.ndarray:
    return cv2.flip(img, code)


def transpose(img: np.ndarray) -> np.ndarray:
    return img.transpose(1, 0, 2) if len(img.shape) > 2 else img.transpose(1, 0)


def rot90(img: np.ndarray, factor: int) -> np.ndarray:
    img = np.rot90(img, factor)
    return np.ascontiguousarray(img)


@use_bboxes_ndarray(return_array=True)
def bboxes_vflip(bboxes: BoxesArray, **kwargs) -> BoxesArray:
    """Flip a batch of bounding boxes vertically around the x-axis.
    Args:
        bboxes (numpy.ndarray): A batch of bounding boxes in `albumentations` format.
        **kwargs:

    Returns:
        numpy.ndarray: A batch of flipped bounding boxes in `albumentations` format.
    """
    if not len(bboxes):
        return bboxes
    bboxes[:, 1::2] = 1 - bboxes[:, -1::-2]
    return bboxes


@use_bboxes_ndarray(return_array=True)
def bboxes_hflip(bboxes: BoxesArray, **kwargs) -> BoxesArray:
    """Flip a batch of bounding boxes horizontally around the y-axis.
    Args:
        bboxes (BoxesArray): A batch of bounding boxes in `albumentations` format.
        **kwargs:

    Returns:
        BoxesArray: A batch of flipped bounding boxes in `albumentations` format.
    """
    if not len(bboxes):
        return bboxes
    bboxes[:, 0::2] = 1 - bboxes[:, -2::-2]
    return bboxes


@ensure_bboxes_format
@use_bboxes_ndarray(return_array=True)
def bboxes_flip(bboxes: BoxesArray, d: int, **kwargs) -> BoxesArray:
    """Flip a batch of bounding boxes either vertically, horizontally or both depending on the value of `d`.

    Args:
        bboxes (numpy.ndarray): A batch of bounding boxes in `albumentations` format.
        d (int): dimension 0 for vertical flip, 1 for horizontal, -1 for transpose.
        **kwargs:

    Returns:
        numpy.ndarray: A batch of flipped bounding boxes in `albumentations` format.

    Raises:
        ValueError: if value of `d` is not -1, 0 or 1.

    """
    if d == 0:
        return bboxes_vflip(bboxes, **kwargs)
    elif d == 1:
        return bboxes_hflip(bboxes, **kwargs)
    elif d == -1:
        bboxes = bboxes_hflip(bboxes, **kwargs)
        return bboxes_vflip(bboxes, **kwargs)
    else:
        raise ValueError(f"Invalid d value {d}. Valid values are -1, 0 and 1.")


@ensure_bboxes_format
@use_bboxes_ndarray(return_array=True)
def bboxes_transpose(bboxes: BoxesArray, axis: int, **kwargs) -> BoxesArray:
    """Transpose bounding bboxes along a given axis in batch.
    Args:
        bboxes (numpy.ndarray): A batch of bounding boxes with `albumentations` format.
        axis (int): 0 as the main axis, 1 as the secondary axis.
        **kwargs:

    Returns:
        numpy.ndarray: A batch of transposed bounding boxes.

    """
    if not len(bboxes):
        return bboxes
    if axis not in {0, 1}:
        raise ValueError(f"Invalid axis value {axis}. Axis must be either 0 or 1")

    if axis == 0:
        bboxes = bboxes[:, [1, 0, 3, 2]]
    else:
        bboxes = 1 - bboxes[:, ::-1]
    return bboxes


@use_keypoints_ndarray(return_array=True)
@angles_2pi_range
def keypoints_vflip(keypoints: KeypointsArray, rows: int, cols: int) -> KeypointsArray:
    """Flip a batch of keypoints vertically around the x-axis.

    Args:
        keypoints (KeypointsArray): A batch of keypoints in `(x, y, angle, scale)` format.
        rows: Image height.
        cols: Image width.

    Returns:
        KeypointsArray: A batch of keypoints `(x, y, angle, scale)`.

    """
    if not len(keypoints):
        return keypoints
    keypoints[..., 2] = -keypoints[..., 2]
    keypoints[..., 1] = rows - 1 - keypoints[..., 1]
    return keypoints


@use_keypoints_ndarray(return_array=True)
@angles_2pi_range
def keypoints_hflip(keypoints: KeypointsArray, rows: int, cols: int) -> KeypointsArray:
    """Flip a batch of keypoints horizontally around the y-axis.

    Args:
        keypoints (KeypointsArray): A batch of keypoints in `(x, y, angle, scale)` format.
        rows: Image height.
        cols: Image width.

    Returns:
        KeypointsArray: A batch of keypoints `(x, y, angle, scale)`.

    """
    if not len(keypoints):
        return keypoints
    keypoints[..., 2] = math.pi - keypoints[..., 2]
    keypoints[..., 0] = cols - 1 - keypoints[..., 0]
    return keypoints


@ensure_keypoints_format
@use_keypoints_ndarray(return_array=True)
def keypoints_flip(keypoints: KeypointsArray, d: int, rows: int, cols: int) -> KeypointsArray:
    """Flip a batch of keypoints either vertically, horizontally or both depending on the value of `d`.

    Args:
        keypoints (KeypointsInternalType): A batch of keypoints in `(x, y, angle, scale)` format.
        d: Number of flip. Must be -1, 0 or 1:
            * 0 - vertical flip,
            * 1 - horizontal flip,
            * -1 - vertical and horizontal flip.
        rows (int): Image height.
        cols (int): Image width.

    Returns:
        KeypointsInternalType: A batch of keypoints `(x, y, angle, scale)`.

    Raises:
        ValueError: if value of `d` is not -1, 0 or 1.

    """
    if d == 0:
        keypoints = keypoints_vflip(keypoints, rows, cols)
    elif d == 1:
        keypoints = keypoints_hflip(keypoints, rows, cols)
    elif d == -1:
        keypoints = keypoints_hflip(keypoints, rows, cols)
        keypoints = keypoints_vflip(keypoints, rows, cols)
    else:
        raise ValueError(f"Invalid d value {d}. Valid values are -1, 0 and 1")
    return keypoints


@ensure_keypoints_format
@use_keypoints_ndarray(return_array=True)
def keypoints_transpose(keypoints: KeypointsArray) -> KeypointsArray:
    """Rotate a batch of keypoints by angle.

    Args:
        keypoints (KeypointsArray): A batch of keypoints in `(x, y, angle, scale)` format.

    Returns:
        KeypointsArray: A batch of keypoints `(x, y, angle, scale)`.

    """
    if not len(keypoints):
        return keypoints

    mask = keypoints[..., 2] <= np.pi
    keypoints[..., 2][mask] = np.pi - keypoints[..., 2][mask]
    keypoints[..., 2][~mask] = 3 * np.pi - keypoints[..., 2][~mask]
    keypoints[..., [0, 1]] = keypoints[..., [1, 0]]
    return keypoints


@preserve_channel_dim
def pad(
    img: np.ndarray,
    min_height: int,
    min_width: int,
    border_mode: int = cv2.BORDER_REFLECT_101,
    value: Optional[ImageColorType] = None,
) -> np.ndarray:
    height, width = img.shape[:2]

    if height < min_height:
        h_pad_top = int((min_height - height) / 2.0)
        h_pad_bottom = min_height - height - h_pad_top
    else:
        h_pad_top = 0
        h_pad_bottom = 0

    if width < min_width:
        w_pad_left = int((min_width - width) / 2.0)
        w_pad_right = min_width - width - w_pad_left
    else:
        w_pad_left = 0
        w_pad_right = 0

    img = pad_with_params(img, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, border_mode, value)

    if img.shape[:2] != (max(min_height, height), max(min_width, width)):
        raise RuntimeError(
            "Invalid result shape. Got: {}. Expected: {}".format(
                img.shape[:2], (max(min_height, height), max(min_width, width))
            )
        )

    return img


@preserve_channel_dim
def pad_with_params(
    img: np.ndarray,
    h_pad_top: int,
    h_pad_bottom: int,
    w_pad_left: int,
    w_pad_right: int,
    border_mode: int = cv2.BORDER_REFLECT_101,
    value: Optional[ImageColorType] = None,
) -> np.ndarray:
    pad_fn = _maybe_process_in_chunks(
        cv2.copyMakeBorder,
        top=h_pad_top,
        bottom=h_pad_bottom,
        left=w_pad_left,
        right=w_pad_right,
        borderType=border_mode,
        value=value,
    )
    return pad_fn(img)


@preserve_shape
def optical_distortion(
    img: np.ndarray,
    k: int = 0,
    dx: int = 0,
    dy: int = 0,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_REFLECT_101,
    value: Optional[ImageColorType] = None,
) -> np.ndarray:
    """Barrel / pincushion distortion. Unconventional augment.

    Reference:
        |  https://stackoverflow.com/questions/6199636/formulas-for-barrel-pincushion-distortion
        |  https://stackoverflow.com/questions/10364201/image-transformation-in-opencv
        |  https://stackoverflow.com/questions/2477774/correcting-fisheye-distortion-programmatically
        |  http://www.coldvision.io/2017/03/02/advanced-lane-finding-using-opencv/
    """
    height, width = img.shape[:2]

    fx = width
    fy = height

    cx = width * 0.5 + dx
    cy = height * 0.5 + dy

    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    distortion = np.array([k, k, 0, 0, 0], dtype=np.float32)
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, distortion, None, None, (width, height), cv2.CV_32FC1)
    return cv2.remap(img, map1, map2, interpolation=interpolation, borderMode=border_mode, borderValue=value)


@preserve_shape
def grid_distortion(
    img: np.ndarray,
    num_steps: int = 10,
    xsteps: Tuple = (),
    ysteps: Tuple = (),
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_REFLECT_101,
    value: Optional[ImageColorType] = None,
) -> np.ndarray:
    """Perform a grid distortion of an input image.

    Reference:
        http://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
    """
    height, width = img.shape[:2]

    x_step = width // num_steps
    xx = np.zeros(width, np.float32)
    prev = 0
    for idx in range(num_steps + 1):
        x = idx * x_step
        start = int(x)
        end = int(x) + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * xsteps[idx]

        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    y_step = height // num_steps
    yy = np.zeros(height, np.float32)
    prev = 0
    for idx in range(num_steps + 1):
        y = idx * y_step
        start = int(y)
        end = int(y) + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step * ysteps[idx]

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    remap_fn = _maybe_process_in_chunks(
        cv2.remap,
        map1=map_x,
        map2=map_y,
        interpolation=interpolation,
        borderMode=border_mode,
        borderValue=value,
    )
    return remap_fn(img)


@preserve_shape
def elastic_transform_approx(
    img: np.ndarray,
    alpha: float,
    sigma: float,
    alpha_affine: float,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_REFLECT_101,
    value: Optional[ImageColorType] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Elastic deformation of images as described in [Simard2003]_ (with modifications for speed).
    Based on https://gist.github.com/ernestum/601cdf56d2b424757de5

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
    """
    height, width = img.shape[:2]

    # Random affine
    center_square = np.array((height, width), dtype=np.float32) // 2
    square_size = min((height, width)) // 3
    alpha = float(alpha)
    sigma = float(sigma)
    alpha_affine = float(alpha_affine)

    pts1 = np.array(
        [
            center_square + square_size,
            [center_square[0] + square_size, center_square[1] - square_size],
            center_square - square_size,
        ],
        dtype=np.float32,
    )
    pts2 = pts1 + random_utils.uniform(-alpha_affine, alpha_affine, size=pts1.shape, random_state=random_state).astype(
        np.float32
    )
    matrix = cv2.getAffineTransform(pts1, pts2)

    warp_fn = _maybe_process_in_chunks(
        cv2.warpAffine,
        M=matrix,
        dsize=(width, height),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=value,
    )
    img = warp_fn(img)

    dx = random_utils.rand(height, width, random_state=random_state).astype(np.float32) * 2 - 1
    cv2.GaussianBlur(dx, (17, 17), sigma, dst=dx)
    dx *= alpha

    dy = random_utils.rand(height, width, random_state=random_state).astype(np.float32) * 2 - 1
    cv2.GaussianBlur(dy, (17, 17), sigma, dst=dy)
    dy *= alpha

    x, y = np.meshgrid(np.arange(width), np.arange(height))

    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)

    remap_fn = _maybe_process_in_chunks(
        cv2.remap,
        map1=map_x,
        map2=map_y,
        interpolation=interpolation,
        borderMode=border_mode,
        borderValue=value,
    )
    return remap_fn(img)
