import cv2
import math
import numpy as np
import skimage.transform

from scipy.ndimage.filters import gaussian_filter

from ..bbox_utils import denormalize_bbox, normalize_bbox
from ..functional import angle_2pi_range, preserve_channel_dim, _maybe_process_in_chunks, preserve_shape, clipped

from typing import Union, List, Sequence, Tuple, Optional


def bbox_rot90(bbox, factor, rows, cols):  # skipcq: PYL-W0613
    """Rotates a bounding box by 90 degrees CCW (see np.rot90)

    Args:
        bbox (tuple): A bounding box tuple (x_min, y_min, x_max, y_max).
        factor (int): Number of CCW rotations. Must be in set {0, 1, 2, 3} See np.rot90.
        rows (int): Image rows.
        cols (int): Image cols.

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
def keypoint_rot90(keypoint, factor, rows, cols, **params):
    """Rotates a keypoint by 90 degrees CCW (see np.rot90)

    Args:
        keypoint (tuple): A keypoint `(x, y, angle, scale)`.
        factor (int): Number of CCW rotations. Must be in range [0;3] See np.rot90.
        rows (int): Image height.
        cols (int): Image width.

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
def rotate(img, angle, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None):
    height, width = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)

    warp_fn = _maybe_process_in_chunks(
        cv2.warpAffine, M=matrix, dsize=(width, height), flags=interpolation, borderMode=border_mode, borderValue=value
    )
    return warp_fn(img)


def bbox_rotate(bbox, angle, rows, cols):
    """Rotates a bounding box by angle degrees.

    Args:
        bbox (tuple): A bounding box `(x_min, y_min, x_max, y_max)`.
        angle (int): Angle of rotation in degrees.
        rows (int): Image rows.
        cols (int): Image cols.

    Returns:
        A bounding box `(x_min, y_min, x_max, y_max)`.

    """
    x_min, y_min, x_max, y_max = bbox[:4]
    scale = cols / float(rows)
    x = np.array([x_min, x_max, x_max, x_min]) - 0.5
    y = np.array([y_min, y_min, y_max, y_max]) - 0.5
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
    matrix = cv2.getRotationMatrix2D(((cols - 1) * 0.5, (rows - 1) * 0.5), angle, 1.0)
    x, y, a, s = keypoint[:4]
    x, y = cv2.transform(np.array([[[x, y]]]), matrix).squeeze()
    return x, y, a + math.radians(angle), s


@preserve_channel_dim
def shift_scale_rotate(
    img, angle, scale, dx, dy, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None
):
    height, width = img.shape[:2]
    center = (width / 2, height / 2)
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
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += dx * width
    matrix[1, 2] += dy * height

    x, y = cv2.transform(np.array([[[x, y]]]), matrix).squeeze()
    angle = a + math.radians(angle)
    scale = s * scale

    return x, y, angle, scale


def bbox_shift_scale_rotate(bbox, angle, scale, dx, dy, rows, cols, **kwargs):  # skipcq: PYL-W0613
    x_min, y_min, x_max, y_max = bbox[:4]
    height, width = rows, cols
    center = (width / 2, height / 2)
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


@preserve_shape
def elastic_transform(
    img,
    alpha,
    sigma,
    alpha_affine,
    interpolation=cv2.INTER_LINEAR,
    border_mode=cv2.BORDER_REFLECT_101,
    value=None,
    random_state=None,
    approximate=False,
    same_dxdy=False,
):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    Based on https://gist.github.com/ernestum/601cdf56d2b424757de5

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(1234)

    height, width = img.shape[:2]

    # Random affine
    center_square = np.float32((height, width)) // 2
    square_size = min((height, width)) // 3
    alpha = float(alpha)
    sigma = float(sigma)
    alpha_affine = float(alpha_affine)

    pts1 = np.float32(
        [
            center_square + square_size,
            [center_square[0] + square_size, center_square[1] - square_size],
            center_square - square_size,
        ]
    )
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    matrix = cv2.getAffineTransform(pts1, pts2)

    warp_fn = _maybe_process_in_chunks(
        cv2.warpAffine, M=matrix, dsize=(width, height), flags=interpolation, borderMode=border_mode, borderValue=value
    )
    img = warp_fn(img)

    if approximate:
        # Approximate computation smooth displacement map with a large enough kernel.
        # On large images (512+) this is approximately 2X times faster
        dx = random_state.rand(height, width).astype(np.float32) * 2 - 1
        cv2.GaussianBlur(dx, (17, 17), sigma, dst=dx)
        dx *= alpha
        if same_dxdy:
            # Speed up even more
            dy = dx
        else:
            dy = random_state.rand(height, width).astype(np.float32) * 2 - 1
            cv2.GaussianBlur(dy, (17, 17), sigma, dst=dy)
            dy *= alpha
    else:
        dx = np.float32(gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma) * alpha)
        if same_dxdy:
            # Speed up
            dy = dx
        else:
            dy = np.float32(gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma) * alpha)

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
def scale(img, scale, interpolation=cv2.INTER_LINEAR):
    height, width = img.shape[:2]
    new_height, new_width = int(height * scale), int(width * scale)
    return resize(img, new_height, new_width, interpolation)


def keypoint_scale(keypoint: Sequence[float], scale_x: float, scale_y: float):
    """Scales a keypoint by scale_x and scale_y.

    Args:
        keypoint (tuple): A keypoint `(x, y, angle, scale)`.
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
def longest_max_size(img, max_size, interpolation):
    return _func_max_size(img, max_size, interpolation, max)


@preserve_channel_dim
def smallest_max_size(img, max_size, interpolation):
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
    bbox: Sequence[float],
    height: int,
    width: int,
    matrix: np.ndarray,
    max_width: int,
    max_height: int,
    keep_size: bool,
):
    x1, y1, x2, y2 = denormalize_bbox(bbox, height, width)

    points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

    x1, y1, x2, y2 = float("inf"), float("inf"), 0, 0
    for pt in points:
        pt = perspective_keypoint(pt.tolist() + [0, 0], height, width, matrix, max_width, max_height, keep_size)
        x, y = pt[:2]
        x = np.clip(x, 0, width if keep_size else max_width)
        y = np.clip(y, 0, height if keep_size else max_height)
        x1 = min(x1, x)
        x2 = max(x2, x)
        y1 = min(y1, y)
        y2 = max(y2, y)

    x = np.clip([x1, x2], 0, width if keep_size else max_width)
    y = np.clip([y1, y2], 0, height if keep_size else max_height)
    return normalize_bbox(
        (x[0], y[0], x[1], y[1]), height if keep_size else max_height, width if keep_size else max_width
    )


def rotation2DMatrixToEulerAngles(matrix: np.ndarray):
    return np.arctan2(matrix[1, 0], matrix[0, 0])


@angle_2pi_range
def perspective_keypoint(
    keypoint: Union[List[int], List[float]],
    height: int,
    width: int,
    matrix: np.ndarray,
    max_width: int,
    max_height: int,
    keep_size: bool,
):
    x, y, angle, scale = keypoint

    keypoint_vector = np.array([x, y], dtype=np.float32).reshape([1, 1, 2])

    x, y = cv2.perspectiveTransform(keypoint_vector, matrix)[0, 0]
    angle += rotation2DMatrixToEulerAngles(matrix[:2, :2])

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
    keypoint: Sequence[float],
    matrix: skimage.transform.ProjectiveTransform,
    scale: dict,
) -> Sequence[float]:
    if _is_identity_matrix(matrix):
        return keypoint

    x, y, a, s = keypoint[:4]
    x, y = skimage.transform.matrix_transform(np.array([[x, y]]), matrix.params).ravel()
    a += rotation2DMatrixToEulerAngles(matrix.params[:2])
    s *= np.max([scale["x"], scale["y"]])
    return x, y, a, s


def bbox_affine(
    bbox: Sequence[float],
    matrix: skimage.transform.ProjectiveTransform,
    rows: int,
    cols: int,
    output_shape: Sequence[int],
) -> Sequence[float]:
    if _is_identity_matrix(matrix):
        return bbox

    x_min, y_min, x_max, y_max = denormalize_bbox(bbox, rows, cols)
    points = np.array(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ]
    )
    points = skimage.transform.matrix_transform(points, matrix.params)
    points[:, 0] = np.clip(points[:, 0], 0, output_shape[1])
    points[:, 1] = np.clip(points[:, 1], 0, output_shape[0])
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])

    return normalize_bbox((x_min, y_min, x_max, y_max), output_shape[0], output_shape[1])


@preserve_channel_dim
def safe_rotate(
    img: np.ndarray,
    angle: int = 0,
    interpolation: int = cv2.INTER_LINEAR,
    value: int = None,
    border_mode: int = cv2.BORDER_REFLECT_101,
):

    old_rows, old_cols = img.shape[:2]

    # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    image_center = (old_cols / 2, old_rows / 2)

    # Rows and columns of the rotated image (not cropped)
    new_rows, new_cols = safe_rotate_enlarged_img_size(angle=angle, rows=old_rows, cols=old_cols)

    # Rotation Matrix
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # Shift the image to create padding
    rotation_mat[0, 2] += new_cols / 2 - image_center[0]
    rotation_mat[1, 2] += new_rows / 2 - image_center[1]

    # CV2 Transformation function
    warp_affine_fn = _maybe_process_in_chunks(
        cv2.warpAffine,
        M=rotation_mat,
        dsize=(new_cols, new_rows),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=value,
    )

    # rotate image with the new bounds
    rotated_img = warp_affine_fn(img)

    # Resize image back to the original size
    resized_img = resize(img=rotated_img, height=old_rows, width=old_cols, interpolation=interpolation)

    return resized_img


def bbox_safe_rotate(bbox, angle, rows, cols):
    old_rows = rows
    old_cols = cols

    # Rows and columns of the rotated image (not cropped)
    new_rows, new_cols = safe_rotate_enlarged_img_size(angle=angle, rows=old_rows, cols=old_cols)

    col_diff = int(np.ceil(abs(new_cols - old_cols) / 2))
    row_diff = int(np.ceil(abs(new_rows - old_rows) / 2))

    # Normalize shifts
    norm_col_shift = col_diff / new_cols
    norm_row_shift = row_diff / new_rows

    # shift bbox
    shifted_bbox = (
        bbox[0] + norm_col_shift,
        bbox[1] + norm_row_shift,
        bbox[2] + norm_col_shift,
        bbox[3] + norm_row_shift,
    )

    rotated_bbox = bbox_rotate(bbox=shifted_bbox, angle=angle, rows=new_rows, cols=new_cols)

    # Bounding boxes are scale invariant, so this does not need to be rescaled to the old size
    return rotated_bbox


def keypoint_safe_rotate(keypoint, angle, rows, cols):
    old_rows = rows
    old_cols = cols

    # Rows and columns of the rotated image (not cropped)
    new_rows, new_cols = safe_rotate_enlarged_img_size(angle=angle, rows=old_rows, cols=old_cols)

    col_diff = int(np.ceil(abs(new_cols - old_cols) / 2))
    row_diff = int(np.ceil(abs(new_rows - old_rows) / 2))

    # Shift keypoint
    shifted_keypoint = (keypoint[0] + col_diff, keypoint[1] + row_diff, keypoint[2], keypoint[3])

    # Rotate keypoint
    rotated_keypoint = keypoint_rotate(shifted_keypoint, angle, rows=new_rows, cols=new_cols)

    # Scale the keypoint
    return keypoint_scale(rotated_keypoint, old_cols / new_cols, old_rows / new_rows)


def safe_rotate_enlarged_img_size(angle: float, rows: int, cols: int):

    deg_angle = abs(angle)

    # The rotation angle
    angle = np.deg2rad(deg_angle % 90)

    # The width of the frame to contain the rotated image
    r_cols = cols * np.cos(angle) + rows * np.sin(angle)

    # The height of the frame to contain the rotated image
    r_rows = cols * np.sin(angle) + rows * np.cos(angle)

    # The above calculations work as is for 0<90 degrees, and for 90<180 the cols and rows are flipped
    if deg_angle > 90:
        return int(r_cols), int(r_rows)
    else:
        return int(r_rows), int(r_cols)


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
    keypoints: Sequence[Sequence[float]], height: int, width: int, inverted: bool = False
) -> np.ndarray:
    """Generate a ``(H,W,N)`` array of distance maps for ``N`` keypoints.

    The ``n``-th distance map contains at every location ``(y, x)`` the
    euclidean distance to the ``n``-th keypoint.

    This function can be used as a helper when augmenting keypoints with a
    method that only supports the augmentation of images.

    Args:
        keypoint (sequence of float): keypoint coordinates
        height (int): image height
        width (int): image width
        inverted (bool): If ``True``, inverted distance maps are returned where each
            distance value d is replaced by ``d/(d+1)``, i.e. the distance
            maps have values in the range ``(0.0, 1.0]`` with ``1.0`` denoting
            exactly the position of the respective keypoint.

    Returns:
        (H,W,N) ndarray
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
    keypoint: Sequence[float],
    matrix: skimage.transform.PiecewiseAffineTransform,
    h: int,
    w: int,
    keypoints_threshold: float,
) -> Tuple[float, float, float, float]:
    x, y, a, s = keypoint
    dist_maps = to_distance_maps([(x, y)], h, w, True)
    dist_maps = piecewise_affine(dist_maps, matrix, 0, "constant", 0)
    x, y = from_distance_maps(dist_maps, True, {"x": -1, "y": -1}, keypoints_threshold)[0]
    return x, y, a, s


def bbox_piecewise_affine(
    bbox: Sequence[float],
    matrix: skimage.transform.PiecewiseAffineTransform,
    h: int,
    w: int,
    keypoints_threshold: float,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = denormalize_bbox(tuple(bbox), h, w)
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
