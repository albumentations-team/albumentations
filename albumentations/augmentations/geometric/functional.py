import math
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import skimage.transform
from scipy.ndimage.filters import gaussian_filter

from albumentations.augmentations.utils import (
    _maybe_process_in_chunks,
    angle_2pi_range,
    clipped,
    preserve_channel_dim,
    preserve_shape,
)

from ... import random_utils
from ...core.bbox_utils import denormalize_bbox, normalize_bbox
from ...core.transforms_interface import (
    BoxInternalType,
    FillValueType,
    ImageColorType,
    KeypointInternalType,
)
from ..crops import functional as FCrop

__all__ = [
    "optical_distortion",
    "elastic_transform_approx",
    "grid_distortion",
    "pad",
    "pad_with_params",
    "bbox_rot90",
    "keypoint_rot90",
    "rotate",
    "bbox_rotate",
    "keypoint_rotate",
    "shift_scale_rotate",
    "keypoint_shift_scale_rotate",
    "bbox_shift_scale_rotate",
    "bboxes_shift_scale_rotate",
    "bboxes_shift_scale_rotate_reflect",
    "elastic_transform",
    "resize",
    "scale",
    "keypoint_scale",
    "py3round",
    "_func_max_size",
    "longest_max_size",
    "smallest_max_size",
    "perspective",
    "perspective_bbox",
    "rotation2DMatrixToEulerAngles",
    "perspective_keypoint",
    "_is_identity_matrix",
    "warp_affine",
    "keypoint_affine",
    "bbox_affine",
    "safe_rotate",
    "bbox_safe_rotate",
    "keypoint_safe_rotate",
    "piecewise_affine",
    "to_distance_maps",
    "from_distance_maps",
    "keypoint_piecewise_affine",
    "bbox_piecewise_affine",
    "bbox_flip",
    "bbox_hflip",
    "bbox_transpose",
    "bbox_vflip",
    "hflip",
    "hflip_cv2",
    "transpose",
    "keypoint_flip",
    "keypoint_hflip",
    "keypoint_transpose",
    "keypoint_vflip",
]


def bbox_rot90(bbox: BoxInternalType, factor: int, rows: int, cols: int) -> BoxInternalType:  # skipcq: PYL-W0613
    """Rotates a bounding box by 90 degrees CCW (see np.rot90)

    Args:
        bbox: A bounding box tuple (x_min, y_min, x_max, y_max).
        factor: Number of CCW rotations. Must be in set {0, 1, 2, 3} See np.rot90.
        rows: Image rows.
        cols: Image cols.

    Returns:
        tuple: A bounding box tuple (x_min, y_min, x_max, y_max).

    """
    if factor not in {0, 1, 2, 3}:
        raise ValueError("Parameter n must be in set {0, 1, 2, 3}")
    x_min, y_min, x_max, y_max = bbox[:4]
    if factor == 1:
        bbox = y_min, 1 - x_max, y_max, 1 - x_min
    elif factor == 2:
        bbox = 1 - x_max, 1 - y_max, 1 - x_min, 1 - y_min
    elif factor == 3:
        bbox = 1 - y_max, x_min, 1 - y_min, x_max
    return bbox


@angle_2pi_range
def keypoint_rot90(keypoint: KeypointInternalType, factor: int, rows: int, cols: int, **params) -> KeypointInternalType:
    """Rotates a keypoint by 90 degrees CCW (see np.rot90)

    Args:
        keypoint: A keypoint `(x, y, angle, scale)`.
        factor: Number of CCW rotations. Must be in range [0;3] See np.rot90.
        rows: Image height.
        cols: Image width.

    Returns:
        tuple: A keypoint `(x, y, angle, scale)`.

    Raises:
        ValueError: if factor not in set {0, 1, 2, 3}

    """
    x, y, angle, scale = keypoint[:4]

    if factor not in {0, 1, 2, 3}:
        raise ValueError("Parameter n must be in set {0, 1, 2, 3}")

    if factor == 1:
        x, y, angle = y, (cols - 1) - x, angle - math.pi / 2
    elif factor == 2:
        x, y, angle = (cols - 1) - x, (rows - 1) - y, angle - math.pi
    elif factor == 3:
        x, y, angle = (rows - 1) - y, x, angle + math.pi / 2

    return x, y, angle, scale


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


def bbox_rotate(bbox: BoxInternalType, angle: float, method: str, rows: int, cols: int) -> BoxInternalType:
    """Rotates a bounding box by angle degrees.

    Args:
        bbox: A bounding box `(x_min, y_min, x_max, y_max)`.
        angle: Angle of rotation in degrees.
        method: Rotation method used. Should be one of: "largest_box", "ellipse". Default: "largest_box".
        rows: Image rows.
        cols: Image cols.

    Returns:
        A bounding box `(x_min, y_min, x_max, y_max)`.

    References:
        https://arxiv.org/abs/2109.13488

    """
    x_min, y_min, x_max, y_max = bbox[:4]
    scale = cols / float(rows)
    if method == "largest_box":
        x = np.array([x_min, x_max, x_max, x_min]) - 0.5
        y = np.array([y_min, y_min, y_max, y_max]) - 0.5
    elif method == "ellipse":
        w = (x_max - x_min) / 2
        h = (y_max - y_min) / 2
        data = np.arange(0, 360, dtype=np.float32)
        x = w * np.sin(np.radians(data)) + (w + x_min - 0.5)
        y = h * np.cos(np.radians(data)) + (h + y_min - 0.5)
    else:
        raise ValueError(f"Method {method} is not a valid rotation method.")
    angle = np.deg2rad(angle)
    x_t = (np.cos(angle) * x * scale + np.sin(angle) * y) / scale
    y_t = -np.sin(angle) * x * scale + np.cos(angle) * y
    x_t = x_t + 0.5
    y_t = y_t + 0.5

    x_min, x_max = min(x_t), max(x_t)
    y_min, y_max = min(y_t), max(y_t)

    return x_min, y_min, x_max, y_max


@angle_2pi_range
def keypoint_rotate(keypoint, angle, rows, cols, **params):
    """Rotate a keypoint by angle.

    Args:
        keypoint (tuple): A keypoint `(x, y, angle, scale)`.
        angle (float): Rotation angle.
        rows (int): Image height.
        cols (int): Image width.

    Returns:
        tuple: A keypoint `(x, y, angle, scale)`.

    """
    center = (cols - 1) * 0.5, (rows - 1) * 0.5
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    x, y, a, s = keypoint[:4]
    x, y = cv2.transform(np.array([[[x, y]]]), matrix).squeeze()
    return x, y, a + math.radians(angle), s


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


@angle_2pi_range
def keypoint_shift_scale_rotate(keypoint, angle, scale, dx, dy, rows, cols, **params):
    (
        x,
        y,
        a,
        s,
    ) = keypoint[:4]
    height, width = rows, cols
    center = (cols - 1) * 0.5, (rows - 1) * 0.5
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += dx * width
    matrix[1, 2] += dy * height

    x, y = cv2.transform(np.array([[[x, y]]]), matrix).squeeze()
    angle = a + math.radians(angle)
    scale = s * scale

    return x, y, angle, scale


def bbox_shift_scale_rotate(bbox, angle, scale, dx, dy, rotate_method, rows, cols, **kwargs):  # skipcq: PYL-W0613
    """Rotates, shifts and scales a bounding box. Rotation is made by angle degrees,
    scaling is made by scale factor and shifting is made by dx and dy.


    Args:
        bbox (tuple): A bounding box `(x_min, y_min, x_max, y_max)`.
        angle (int): Angle of rotation in degrees.
        scale (int): Scale factor.
        dx (int): Shift along x-axis in pixel units.
        dy (int): Shift along y-axis in pixel units.
        rotate_method(str): Rotation method used. Should be one of: "largest_box", "ellipse".
            Default: "largest_box".
        rows (int): Image rows.
        cols (int): Image cols.

    Returns:
        A bounding box `(x_min, y_min, x_max, y_max)`.

    """
    height, width = rows, cols
    center = (width / 2, height / 2)
    if rotate_method == "ellipse":
        x_min, y_min, x_max, y_max = bbox_rotate(bbox, angle, rotate_method, rows, cols)
        matrix = cv2.getRotationMatrix2D(center, 0, scale)
    else:
        x_min, y_min, x_max, y_max = bbox[:4]
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += dx * width
    matrix[1, 2] += dy * height
    x = np.array([x_min, x_max, x_max, x_min])
    y = np.array([y_min, y_min, y_max, y_max])
    ones = np.ones(shape=(len(x)))
    points_ones = np.vstack([x, y, ones]).transpose()
    points_ones[:, 0] *= width
    points_ones[:, 1] *= height
    tr_points = matrix.dot(points_ones.T).T
    tr_points[:, 0] /= width
    tr_points[:, 1] /= height

    x_min, x_max = min(tr_points[:, 0]), max(tr_points[:, 0])
    y_min, y_max = min(tr_points[:, 1]), max(tr_points[:, 1])

    return x_min, y_min, x_max, y_max


def bboxes_shift_scale_rotate(bboxes, angle, scale, dx, dy, rotate_method, rows, cols, **kwargs):
    """(numpy version of bbox_shift_scale_rotate)"""
    if not isinstance(bboxes, np.ndarray):
        raise ValueError("bboxes should be np.ndarray")
    n = bboxes.shape[0]

    height, width = rows, cols
    center = (width / 2, height / 2)

    if rotate_method == "ellipse":
        bboxes = np.array(
            [bbox_rotate(bbox[:4], angle, rotate_method, rows, cols) + tuple(bbox[4:]) for bbox in bboxes]
        )
        matrix = cv2.getRotationMatrix2D(center, 0, scale)
    else:
        matrix = cv2.getRotationMatrix2D(center, angle, scale)

    matrix[0, 2] += dx * width
    matrix[1, 2] += dy * height

    # bboxes to point_onees vector: [x, y, 1]
    x1y1 = bboxes[:, [0, 1]]
    x2y1 = bboxes[:, [2, 1]]
    x2y2 = bboxes[:, [2, 3]]
    x1y2 = bboxes[:, [0, 3]]

    points_ones = np.empty((4 * n, 3), dtype=bboxes.dtype)
    points_ones[0 * n : 1 * n, :2] = x1y1
    points_ones[1 * n : 2 * n, :2] = x2y1
    points_ones[2 * n : 3 * n, :2] = x2y2
    points_ones[3 * n : 4 * n, :2] = x1y2
    points_ones[:, 2] = 1

    # denormalize
    points_ones[:, 0] *= width
    points_ones[:, 1] *= height

    # transform
    tr_points = matrix.dot(points_ones.T).T

    # normalize
    tr_points[:, 0] /= width
    tr_points[:, 1] /= height

    # points to bboxes
    xs = tr_points.reshape((4, n, 2))[:, :, 0].T
    ys = tr_points.reshape((4, n, 2))[:, :, 1].T
    tr_bboxes = bboxes.copy()
    tr_bboxes[:, 0] = xs.min(axis=1)
    tr_bboxes[:, 1] = ys.min(axis=1)
    tr_bboxes[:, 2] = xs.max(axis=1)
    tr_bboxes[:, 3] = ys.max(axis=1)

    return tr_bboxes


def _estimate_expand_grid_size(rows, cols, scale=1):
    """Estimate the number of grid cells to cover the reachable area (This may be overestimate)."""
    cell_size = max(rows, cols)
    d_x = scale * cols
    d_y = scale * rows

    # Estimate the distance of the point that is farthest away from the center point
    # by thinking about the case of 1.0 translation and 45 degree rotation.
    n_x = 1 + 2 * int(np.ceil((cell_size * (1 + np.sqrt(2) / 2) - d_x / 2) / d_x))
    n_y = 1 + 2 * int(np.ceil((cell_size * (1 + np.sqrt(2) / 2) - d_y / 2) / d_y))
    return n_x, n_y


def bboxes_expand_grid(bboxes, n_x, n_y, rows, cols, border_mode=cv2.BORDER_WRAP):
    """Make n_x by n_y grid from copies of bounding boxes with centering the original cell.
    The border type is taken into account when copies are layedout.
    Args:
        bboxes (np.ndarray): A two dimensional ndarray. Each row is `x_min, y_min, x_max, y_max` or
        `x_min, y_min, x_max, y_max, label_index`.
        n_x (int): A number of grid cell in x-axis. Should be odd number.
        n_y (int): A number of grid cell in y-axis. Should be odd number.
        rows (int): Image rows.
        cols (int): Image cols.
        border_model (int): Border model. Should be one of:
        `cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101`.
    Returns:
        bboxes (np.ndarray): A two dimensional ndarray. Each row is `x_min, y_min, x_max, y_max` or
        `x_min, y_min, x_max, y_max, label_index`.


    """
    if not isinstance(bboxes, np.ndarray):
        raise ValueError("bboxes should be np.ndarray")

    if n_x <= 0 or n_y <= 0:
        raise ValueError(f"n_x and n_y should be non zero positive numbers. got {n_x, n_y}")

    if n_x % 2 == 0 and n_y % 2 == 0:
        raise ValueError(f"n_x and n_y should be odd numbers. got {n_x, n_y}")

    if border_mode not in [cv2.BORDER_REFLECT, cv2.BORDER_REFLECT_101, cv2.BORDER_WRAP]:
        raise ValueError(
            f"Select border_mode from "
            f"[cv2.BORDER_REFLECT({cv2.BORDER_REFLECT}), cv2.BORDER_WRAP({cv2.BORDER_WRAP}),"
            f" cv2.BORDER_REFLECT_101({cv2.BORDER_REFLECT_101})], got {border_mode}"
        )

    nb, n_coord = bboxes.shape[:2]

    if n_coord not in [4, 5]:
        raise ValueError(f"The number of bounding box elements should be 4 or 5. got {n_coord}")

    # Coordinates of the bboxes in the (i, j)-th grid cell from the top-left origin are given by bboxes + (j, i, j, i).
    # Pre-compute "shift" matrix for later use. (fifth elements are dummy value that should be zero)
    # if nb == 1 the shift have the following values:
    #
    # (0, 0, 0, 0)  (1, 0, 1, 0) (2, 0, 2, 0) ...
    # (0, 1, 0, 1)  (1, 1, 1, 1) (2, 1, 2, 1) ...
    # (0, 2, 0, 2)  (1, 2, 1, 2) (2, 2, 2, 1) ...
    # ...
    #
    shift = np.indices([n_x, n_y]).transpose(2, 1, 0)
    if n_coord == 4:
        shift = np.concatenate([shift, shift], axis=2)
    else:  # n_coord == 5
        shift = np.concatenate([shift, shift, np.zeros((n_y, n_x, 1))], axis=2)

    # [grid rows, grid cols, number of box, bbox coordinates]
    shift = shift.reshape((n_y, n_x, 1, n_coord))

    if border_mode in [cv2.BORDER_REFLECT, cv2.BORDER_REFLECT_101]:
        # With n_x=3, n_y=3 and the input "bbox" desplayed by "b", the resulting output will be:
        #
        #       q p q
        #  b -> d b d
        #       q p q
        #
        # where, b: original, d: h_flip, p: v_flip, q: h_flip and v_flip

        # Pre-calculate all flipped bboxes patterns
        flip_bbox_map = np.zeros((2, 2, nb, n_coord), dtype=type(bboxes[0][0]))
        flip_bbox_map[0, 0] = np.array(bboxes).copy()
        flip_bbox_map[0, 1] = bboxes_hflip(bboxes, rows, cols)
        flip_bbox_map[1, 0] = bboxes_vflip(bboxes, rows, cols)
        flip_bbox_map[1, 1] = bboxes_flip(bboxes, -1, rows, cols)

        # The relation between flip and flags are:
        #
        #  q p q       (1, 1)(1, 0)(1, 1)
        #  d b d  <->  (0, 1)(0, 0)(0, 1)
        #  q p q       (1, 1)(1, 0)(1, 1)
        #
        flag_y0 = (n_y // 2) % 2
        flag_x0 = (n_x // 2) % 2
        flags_y, flags_x = np.indices((n_y, n_x)) % 2
        flags_y = (flag_y0 + flags_y) % 2
        flags_x = (flag_x0 + flags_x) % 2
        # Expand copies of bboxes over the grid
        bboxes_exp = flip_bbox_map[flags_y, flags_x] + shift

    elif border_mode in [cv2.BORDER_WRAP]:
        # with n=3 and the input "bbox" desplayed by "b", the resulting output will be:
        #
        #       b b b
        #  b -> b b b
        #       b b b
        #
        bboxes_exp = np.repeat(np.repeat(bboxes.reshape(1, 1, nb, n_coord), n_y, axis=0), n_x, axis=1) + shift

    # normalize
    grid_sizes = [n_x, n_y, n_x, n_y]
    if n_coord == 5:
        grid_sizes += [1]
    bboxes_exp = bboxes_exp.reshape(-1, n_coord) / np.array(grid_sizes).reshape(1, n_coord)

    return bboxes_exp


def bboxes_shift_scale_rotate_reflect(
    bboxes, angle, scale, dx, dy, rotate_method, rows, cols, border_mode=cv2.BORDER_REFLECT_101
):

    if not isinstance(bboxes, np.ndarray):
        raise ValueError("bboxes should be np.ndarray")

    # Make n_x by n_y grid layout.
    # ex. n_x = n_y = 3, border = cv2.BORDER_REFLECT
    #        +-+-+-+
    #        |q|p|q|
    #        +-+-+-+
    #   b -> |d|b|d|
    #        +-+-+-+
    #        |q|p|q|
    #        +-+-+-+
    if border_mode in [cv2.BORDER_REFLECT_101, cv2.BORDER_WRAP, cv2.BORDER_REFLECT]:
        # Estimate the number of grid cells to cover the reachable area.
        # Current implementation expect that n_x and n_y are odd numbers.
        n_x, n_y = _estimate_expand_grid_size(rows, cols, scale)
        m_x, m_y = n_x // 2, n_y // 2  # grid index of the center cell

        # Expand bboxes over the whole grid cells
        bboxes_exp = bboxes_expand_grid(bboxes, n_x, n_y, rows, cols, border_mode=border_mode)
    else:
        n_x, n_y = 1, 1
        m_x, m_y = 0, 0
        bboxes_exp = bboxes

    # Apply affine transform. Note that the area is magnified by n_x and n_y, the dx and dy are rescaled.
    # ex. (dx, dy) = (1, 1)
    #                    q p q
    #    +-+-+-+      +-+-+-+
    #    |q|p|q|      | |d|b|d
    #    +-+-+-+      +-+-+-+
    #    |d|b|d|  ->  | |q|p|q
    #    +-+-+-+      +-+-+-+
    #    |q|p|q|      | | | |
    #    +-+-+-+      +-+-+-+
    tr_bboxes = bboxes_shift_scale_rotate(
        bboxes_exp, angle, scale, dx / n_x, dy / n_y, rotate_method, n_y * rows, n_x * cols
    )
    # Crop the center cell from the grid, and remove boxes outside the cropped cell.
    #      q p q
    #   +-+-+-+
    #   | |d|b|d
    #   +-+-+-+       +-+
    #   | |q|p|q  ->  |q|
    #   +-+-+-+       +-+
    #   | | | |
    #   +-+-+-+
    crop_coords = [m_x * cols, m_y * rows, (m_x + 1) * cols, (m_y + 1) * rows]
    bboxes_crop = FCrop.crop_bboxes_by_coords(tr_bboxes, crop_coords, rows, cols, n_y * rows, n_x * cols)
    # Remove bboxes that reside outside the cropped cell.
    x_out = np.logical_or(bboxes_crop[:, [0, 2]].max(axis=1) <= 0, bboxes_crop[:, [0, 2]].min(axis=1) >= 1)
    y_out = np.logical_or(bboxes_crop[:, [1, 3]].max(axis=1) <= 0, bboxes_crop[:, [1, 3]].min(axis=1) >= 1)
    is_outside = np.logical_or(x_out, y_out)
    is_inside = np.logical_not(is_outside)
    bboxes_crop = bboxes_crop[is_inside, :]
    return bboxes_crop


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


def keypoint_scale(keypoint: KeypointInternalType, scale_x: float, scale_y: float) -> KeypointInternalType:
    """Scales a keypoint by scale_x and scale_y.

    Args:
        keypoint: A keypoint `(x, y, angle, scale)`.
        scale_x: Scale coefficient x-axis.
        scale_y: Scale coefficient y-axis.

    Returns:
        A keypoint `(x, y, angle, scale)`.

    """
    x, y, angle, scale = keypoint[:4]
    return x * scale_x, y * scale_y, angle, scale * max(scale_x, scale_y)


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


def perspective_bbox(
    bbox: BoxInternalType,
    height: int,
    width: int,
    matrix: np.ndarray,
    max_width: int,
    max_height: int,
    keep_size: bool,
) -> BoxInternalType:
    x1, y1, x2, y2 = denormalize_bbox(bbox, height, width)[:4]

    points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

    x1, y1, x2, y2 = float("inf"), float("inf"), 0, 0
    for pt in points:
        pt = perspective_keypoint(pt.tolist() + [0, 0], height, width, matrix, max_width, max_height, keep_size)
        x, y = pt[:2]
        x1 = min(x1, x)
        x2 = max(x2, x)
        y1 = min(y1, y)
        y2 = max(y2, y)

    return normalize_bbox((x1, y1, x2, y2), height if keep_size else max_height, width if keep_size else max_width)


def rotation2DMatrixToEulerAngles(matrix: np.ndarray, y_up: bool = False) -> float:
    """
    Args:
        matrix (np.ndarray): Rotation matrix
        y_up (bool): is Y axis looks up or down
    """
    if y_up:
        return np.arctan2(matrix[1, 0], matrix[0, 0])
    return np.arctan2(-matrix[1, 0], matrix[0, 0])


@angle_2pi_range
def perspective_keypoint(
    keypoint: KeypointInternalType,
    height: int,
    width: int,
    matrix: np.ndarray,
    max_width: int,
    max_height: int,
    keep_size: bool,
) -> KeypointInternalType:
    x, y, angle, scale = keypoint

    keypoint_vector = np.array([x, y], dtype=np.float32).reshape([1, 1, 2])

    x, y = cv2.perspectiveTransform(keypoint_vector, matrix)[0, 0]
    angle += rotation2DMatrixToEulerAngles(matrix[:2, :2], y_up=True)

    scale_x = np.sign(matrix[0, 0]) * np.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2)
    scale_y = np.sign(matrix[1, 1]) * np.sqrt(matrix[1, 0] ** 2 + matrix[1, 1] ** 2)
    scale *= max(scale_x, scale_y)

    if keep_size:
        scale_x = width / max_width
        scale_y = height / max_height
        return keypoint_scale((x, y, angle, scale), scale_x, scale_y)

    return x, y, angle, scale


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


@angle_2pi_range
def keypoint_affine(
    keypoint: KeypointInternalType,
    matrix: skimage.transform.ProjectiveTransform,
    scale: dict,
) -> KeypointInternalType:
    if _is_identity_matrix(matrix):
        return keypoint

    x, y, a, s = keypoint[:4]
    x, y = cv2.transform(np.array([[[x, y]]]), matrix.params[:2]).squeeze()
    a += rotation2DMatrixToEulerAngles(matrix.params[:2])
    s *= np.max([scale["x"], scale["y"]])
    return x, y, a, s


def bbox_affine(
    bbox: BoxInternalType,
    matrix: skimage.transform.ProjectiveTransform,
    rows: int,
    cols: int,
    output_shape: Sequence[int],
) -> BoxInternalType:
    if _is_identity_matrix(matrix):
        return bbox

    x_min, y_min, x_max, y_max = denormalize_bbox(bbox, rows, cols)[:4]
    points = np.array(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ]
    )
    points = skimage.transform.matrix_transform(points, matrix.params)
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])

    return normalize_bbox((x_min, y_min, x_max, y_max), output_shape[0], output_shape[1])


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


def bbox_safe_rotate(bbox: BoxInternalType, matrix: np.ndarray, cols: int, rows: int) -> BoxInternalType:
    x1, y1, x2, y2 = denormalize_bbox(bbox, rows, cols)[:4]
    points = np.array(
        [
            [x1, y1, 1],
            [x2, y1, 1],
            [x2, y2, 1],
            [x1, y2, 1],
        ]
    )
    points = points @ matrix.T
    x1 = points[:, 0].min()
    x2 = points[:, 0].max()
    y1 = points[:, 1].min()
    y2 = points[:, 1].max()

    def fix_point(pt1: float, pt2: float, max_val: float) -> Tuple[float, float]:
        # In my opinion, these errors should be very low, around 1-2 pixels.
        if pt1 < 0:
            return 0, pt2 + pt1
        if pt2 > max_val:
            return pt1 - (pt2 - max_val), max_val
        return pt1, pt2

    x1, x2 = fix_point(x1, x2, cols)
    y1, y2 = fix_point(y1, y2, rows)

    return normalize_bbox((x1, y1, x2, y2), rows, cols)


def keypoint_safe_rotate(
    keypoint: KeypointInternalType,
    matrix: np.ndarray,
    angle: float,
    scale_x: float,
    scale_y: float,
    cols: int,
    rows: int,
) -> KeypointInternalType:
    x, y, a, s = keypoint[:4]
    point = np.array([[x, y, 1]])
    x, y = (point @ matrix.T)[0]

    # To avoid problems with float errors
    x = np.clip(x, 0, cols - 1)
    y = np.clip(y, 0, rows - 1)

    a += angle
    s *= max(scale_x, scale_y)
    return x, y, a, s


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
    keypoints: Sequence[Tuple[float, float]], height: int, width: int, inverted: bool = False
) -> np.ndarray:
    """Generate a ``(H,W,N)`` array of distance maps for ``N`` keypoints.

    The ``n``-th distance map contains at every location ``(y, x)`` the
    euclidean distance to the ``n``-th keypoint.

    This function can be used as a helper when augmenting keypoints with a
    method that only supports the augmentation of images.

    Args:
        keypoint: keypoint coordinates
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


def keypoint_piecewise_affine(
    keypoint: KeypointInternalType,
    matrix: skimage.transform.PiecewiseAffineTransform,
    h: int,
    w: int,
    keypoints_threshold: float,
) -> KeypointInternalType:
    x, y, a, s = keypoint[:4]
    dist_maps = to_distance_maps([(x, y)], h, w, True)
    dist_maps = piecewise_affine(dist_maps, matrix, 0, "constant", 0)
    x, y = from_distance_maps(dist_maps, True, {"x": -1, "y": -1}, keypoints_threshold)[0]
    return x, y, a, s


def bbox_piecewise_affine(
    bbox: BoxInternalType,
    matrix: skimage.transform.PiecewiseAffineTransform,
    h: int,
    w: int,
    keypoints_threshold: float,
) -> BoxInternalType:
    x1, y1, x2, y2 = denormalize_bbox(bbox, h, w)[:4]
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
    return normalize_bbox((x1, y1, x2, y2), h, w)


def vflip(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[::-1, ...])


def hflip(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[:, ::-1, ...])


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


def bbox_vflip(bbox: BoxInternalType, rows: int, cols: int) -> BoxInternalType:  # skipcq: PYL-W0613
    """Flip a bounding box vertically around the x-axis.

    Args:
        bbox: A bounding box `(x_min, y_min, x_max, y_max)`.
        rows: Image rows.
        cols: Image cols.

    Returns:
        tuple: A bounding box `(x_min, y_min, x_max, y_max)`.

    """
    x_min, y_min, x_max, y_max = bbox[:4]
    return x_min, 1 - y_max, x_max, 1 - y_min


def bbox_hflip(bbox: BoxInternalType, rows: int, cols: int) -> BoxInternalType:  # skipcq: PYL-W0613
    """Flip a bounding box horizontally around the y-axis.

    Args:
        bbox: A bounding box `(x_min, y_min, x_max, y_max)`.
        rows: Image rows.
        cols: Image cols.

    Returns:
        A bounding box `(x_min, y_min, x_max, y_max)`.

    """
    x_min, y_min, x_max, y_max = bbox[:4]
    return 1 - x_max, y_min, 1 - x_min, y_max


def bbox_flip(bbox: BoxInternalType, d: int, rows: int, cols: int) -> BoxInternalType:
    """Flip a bounding box either vertically, horizontally or both depending on the value of `d`.

    Args:
        bbox: A bounding box `(x_min, y_min, x_max, y_max)`.
        d: dimension. 0 for vertical flip, 1 for horizontal, -1 for transpose
        rows: Image rows.
        cols: Image cols.

    Returns:
        A bounding box `(x_min, y_min, x_max, y_max)`.

    Raises:
        ValueError: if value of `d` is not -1, 0 or 1.

    """
    if d == 0:
        bbox = bbox_vflip(bbox, rows, cols)
    elif d == 1:
        bbox = bbox_hflip(bbox, rows, cols)
    elif d == -1:
        bbox = bbox_hflip(bbox, rows, cols)
        bbox = bbox_vflip(bbox, rows, cols)
    else:
        raise ValueError("Invalid d value {}. Valid values are -1, 0 and 1".format(d))
    return bbox


def bbox_transpose(
    bbox: KeypointInternalType, axis: int, rows: int, cols: int
) -> KeypointInternalType:  # skipcq: PYL-W0613
    """Transposes a bounding box along given axis.

    Args:
        bbox: A bounding box `(x_min, y_min, x_max, y_max)`.
        axis: 0 - main axis, 1 - secondary axis.
        rows: Image rows.
        cols: Image cols.

    Returns:
        A bounding box tuple `(x_min, y_min, x_max, y_max)`.

    Raises:
        ValueError: If axis not equal to 0 or 1.

    """
    x_min, y_min, x_max, y_max = bbox[:4]
    if axis not in {0, 1}:
        raise ValueError("Axis must be either 0 or 1.")
    if axis == 0:
        bbox = (y_min, x_min, y_max, x_max)
    if axis == 1:
        bbox = (1 - y_max, 1 - x_max, 1 - y_min, 1 - x_min)
    return bbox


@angle_2pi_range
def keypoint_vflip(keypoint: KeypointInternalType, rows: int, cols: int) -> KeypointInternalType:
    """Flip a keypoint vertically around the x-axis.

    Args:
        keypoint: A keypoint `(x, y, angle, scale)`.
        rows: Image height.
        cols: Image width.

    Returns:
        tuple: A keypoint `(x, y, angle, scale)`.

    """
    x, y, angle, scale = keypoint[:4]
    angle = -angle
    return x, (rows - 1) - y, angle, scale


@angle_2pi_range
def keypoint_hflip(keypoint: KeypointInternalType, rows: int, cols: int) -> KeypointInternalType:
    """Flip a keypoint horizontally around the y-axis.

    Args:
        keypoint: A keypoint `(x, y, angle, scale)`.
        rows: Image height.
        cols: Image width.

    Returns:
        A keypoint `(x, y, angle, scale)`.

    """
    x, y, angle, scale = keypoint[:4]
    angle = math.pi - angle
    return (cols - 1) - x, y, angle, scale


def keypoint_flip(keypoint: KeypointInternalType, d: int, rows: int, cols: int) -> KeypointInternalType:
    """Flip a keypoint either vertically, horizontally or both depending on the value of `d`.

    Args:
        keypoint: A keypoint `(x, y, angle, scale)`.
        d: Number of flip. Must be -1, 0 or 1:
            * 0 - vertical flip,
            * 1 - horizontal flip,
            * -1 - vertical and horizontal flip.
        rows: Image height.
        cols: Image width.

    Returns:
        A keypoint `(x, y, angle, scale)`.

    Raises:
        ValueError: if value of `d` is not -1, 0 or 1.

    """
    if d == 0:
        keypoint = keypoint_vflip(keypoint, rows, cols)
    elif d == 1:
        keypoint = keypoint_hflip(keypoint, rows, cols)
    elif d == -1:
        keypoint = keypoint_hflip(keypoint, rows, cols)
        keypoint = keypoint_vflip(keypoint, rows, cols)
    else:
        raise ValueError(f"Invalid d value {d}. Valid values are -1, 0 and 1")
    return keypoint


def keypoint_transpose(keypoint: KeypointInternalType) -> KeypointInternalType:
    """Rotate a keypoint by angle.

    Args:
        keypoint: A keypoint `(x, y, angle, scale)`.

    Returns:
        A keypoint `(x, y, angle, scale)`.

    """
    x, y, angle, scale = keypoint[:4]

    if angle <= np.pi:
        angle = np.pi - angle
    else:
        angle = 3 * np.pi - angle

    return y, x, angle, scale


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


def bboxes_vflip(bboxes, rows, cols):  # skipcq: PYL-W0613
    """Flip a bounding box vertically around the x-axis. (numpy version of bbox_vflip)
    Args:
        bboxes (np.ndarray): A two dimensional ndarray. Each row is `x_min, y_min, x_max, y_max` or
        `x_min, y_min, x_max, y_max, label_index`.
        rows (int): Image rows.
        cols (int): Image cols.
    Returns:
        bboxes (np.ndarray): A two dimensional ndarray. Each row is `x_min, y_min, x_max, y_max` or
        `x_min, y_min, x_max, y_max, label_index`.
    """
    if not isinstance(bboxes, np.ndarray):
        raise ValueError("bboxes should be np.ndarray")

    new_bboxes = bboxes.copy()
    new_bboxes[:, 0] = bboxes[:, 0]
    new_bboxes[:, 1] = 1 - bboxes[:, 3]  # 1 - y_min
    new_bboxes[:, 2] = bboxes[:, 2]
    new_bboxes[:, 3] = 1 - bboxes[:, 1]  # 1 - y_max

    return new_bboxes


def bboxes_hflip(bboxes, rows, cols):  # skipcq: PYL-W0613
    """Flip a bounding box horizontally around the y-axis. (numpy version of bbox_hflip)
    Args:
        bboxes (np.ndarray): A two dimensional ndarray. Each row is `x_min, y_min, x_max, y_max` or
        `x_min, y_min, x_max, y_max, label_index`.
        rows (int): Image rows.
        cols (int): Image cols.
    Returns:
        bboxes (np.ndarray): A two dimensional ndarray. Each row is `x_min, y_min, x_max, y_max` or
        `x_min, y_min, x_max, y_max, label_index`.
    """
    if not isinstance(bboxes, np.ndarray):
        raise ValueError("bboxes should be np.ndarray")

    new_bboxes = bboxes.copy()
    new_bboxes[:, 0] = 1 - bboxes[:, 2]  # 1 - x_max
    new_bboxes[:, 1] = bboxes[:, 1]
    new_bboxes[:, 2] = 1 - bboxes[:, 0]  # 1 - x_min
    new_bboxes[:, 3] = bboxes[:, 3]

    return new_bboxes


def bboxes_flip(bboxes, d, rows, cols):
    """Flip a bounding box either vertically, horizontally or both depending on the value of `d`.
    Args:
        bboxes (np.ndarray): A two dimensional ndarray. Each row is `x_min, y_min, x_max, y_max` or
        `x_min, y_min, x_max, y_max, label_index`.
        d (int):
        rows (int): Image rows.
        cols (int): Image cols.
    Returns:
        bboxes (np.ndarray): A two dimensional ndarray. Each row is `x_min, y_min, x_max, y_max` or
        `x_min, y_min, x_max, y_max, label_index`.
    Raises:
        ValueError: if value of `d` is not -1, 0 or 1.
    """
    if d == 0:
        bboxes = bboxes_vflip(bboxes, rows, cols)
    elif d == 1:
        bboxes = bboxes_hflip(bboxes, rows, cols)
    elif d == -1:
        bboxes = bboxes_hflip(bboxes, rows, cols)
        bboxes = bboxes_vflip(bboxes, rows, cols)
    else:
        raise ValueError("Invalid d value {}. Valid values are -1, 0 and 1".format(d))
    return bboxes
