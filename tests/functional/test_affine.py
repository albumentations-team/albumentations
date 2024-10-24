import math
from itertools import product

import cv2
import numpy as np
import pytest

import albumentations as A
import albumentations.augmentations.geometric.functional as fgeometric
from albumentations.core.bbox_utils import denormalize_bboxes, normalize_bboxes
from tests.conftest import SQUARE_UINT8_IMAGE

# Define your parameter sets
image_shapes = [
    (100, 100),
    (100, 100, 1),
    (100, 100, 2),
    (100, 100, 3),
    (100, 100, 7),
]

transformation_params = [
    (0, (0, 0), 1, (0, 0), (100, 100)),  # No change
    (45, (0, 0), 1, (0, 0), (100, 100)),  # Rotation
    (0, (10, 10), 1, (0, 0), (100, 100)),  # Translation
    (0, (0, 0), 2, (0, 0), (200, 200)),  # Scaling
    (0, (0, 0), 1, (20, 0), (100, 100)),  # Shear in x only
    (0, (0, 0), 1, (0, 20), (100, 100)),  # Shear in y only
    (0, (0, 0), 1, (20, 20), (100, 100)),  # Shear in both x and y
]

# Combine the two sets of parameters
combined_params = list(product(transformation_params, image_shapes))


@pytest.mark.parametrize("params,image_shape", combined_params)
def test_warp_affine(params, image_shape):
    angle, translation, scale, shear, output_shape = params
    # Create an image of the specified shape
    image = np.ones(image_shape, dtype=np.uint8) * 255

    # Prepare parameters for create_affine_transformation_matrix
    translate: fgeometric.TranslateDict = {"x": translation[0], "y": translation[1]}
    shear_dict: fgeometric.ShearDict = {"x": shear[0], "y": shear[1]}  # Assuming shear is only in x direction
    scale_dict: fgeometric.ScaleDict = {"x": scale, "y": scale}
    shift = fgeometric.center(image_shape)  # Center of the image

    # Create affine transformation matrix
    affine_matrix = fgeometric.create_affine_transformation_matrix(
        translate=translate,
        shear=shear_dict,
        scale=scale_dict,
        rotate=angle,
        shift=shift
    )

    assert affine_matrix.shape == (3, 3), "Affine matrix should be 3x3"

    warped_img = fgeometric.warp_affine(
        image,
        affine_matrix,
        cv2.INTER_LINEAR,
        0,
        cv2.BORDER_CONSTANT,
        output_shape,
    )

    assert warped_img.shape[:2] == output_shape, "Output shape does not match the expected shape."


def create_test_image(shape):
    """Creates a test image with distinct values in each channel."""
    assert len(shape) == 3 and shape[2] >= 2
    image = np.zeros(shape, dtype=np.uint8)

    for channel in range(shape[2]):
        image[:, :, channel] = np.uint8(channel + 1)
    return image


@pytest.mark.parametrize("translate, expected", [
    ({"x": 10, "y": 20}, np.float32([[1, 0, 10], [0, 1, 20]])),
    ({"x": -5, "y": 15}, np.float32([[1, 0, -5], [0, 1, 15]])),
])
def test_translation(translate, expected):
    affine_matrix = fgeometric.create_affine_transformation_matrix(
        translate=translate,
        shear={"x": 0, "y": 0},
        scale={"x": 1, "y": 1},
        rotate=0,
        shift=(0, 0)
    )

    assert affine_matrix.shape == (3, 3), "Affine matrix should be 3x3"
    np.testing.assert_array_almost_equal(affine_matrix[:2], expected)

@pytest.mark.parametrize("scale, expected", [
    ({"x": 2, "y": 2}, np.float32([[2, 0, 0], [0, 2, 0]])),
    ({"x": 0.5, "y": 1.5}, np.float32([[0.5, 0, 0], [0, 1.5, 0]])),
])
def test_scale(scale, expected):
    affine_matrix = fgeometric.create_affine_transformation_matrix(
        translate={"x": 0, "y": 0},
        shear={"x": 0, "y": 0},
        scale=scale,
        rotate=0,
        shift=(0, 0)
    )

    assert affine_matrix.shape == (3, 3), "Affine matrix should be 3x3"
    np.testing.assert_array_almost_equal(affine_matrix[:2], expected)

@pytest.mark.parametrize("rotate", [0, 45, 90, 180, -30])
def test_rotation(rotate):
    result = fgeometric.create_affine_transformation_matrix(
        translate={"x": 0, "y": 0},
        shear={"x": 0, "y": 0},
        scale={"x": 1, "y": 1},
        rotate=rotate,
        shift=(0, 0)
    )
    opencv_result = cv2.getRotationMatrix2D((0, 0), rotate, 1.0)
    np.testing.assert_array_almost_equal(result[:2], opencv_result)

@pytest.mark.parametrize("shear", [
    {"x": 0, "y": 0},
    {"x": 30, "y": 0},
    {"x": 0, "y": 30},
    {"x": 15, "y": 15},
])
def test_shear(shear):
    result = fgeometric.create_affine_transformation_matrix(
        translate={"x": 0, "y": 0},
        shear=shear,
        scale={"x": 1, "y": 1},
        rotate=0,
        shift=(0, 0)
    )
    # OpenCV doesn't have a direct shear function, so we'll just check if the matrix is as expected
    expected = np.array([
        [1, np.tan(np.deg2rad(shear["x"])), 0],
        [np.tan(np.deg2rad(shear["y"])), 1, 0],
        [0, 0, 1]
    ])
        # Compare only the first two rows and columns
    np.testing.assert_array_almost_equal(result[:2, :2], expected[:2, :2])

    # Check the translation part separately
    np.testing.assert_array_almost_equal(result[:2, 2], expected[:2, 2])


@pytest.mark.parametrize("shift", [(10, 20), (-5, 15), (0, 0)])
def test_shift(shift):
    result = fgeometric.create_affine_transformation_matrix(
        translate={"x": 0, "y": 0},
        shear={"x": 0, "y": 0},
        scale={"x": 1, "y": 1},
        rotate=0,
        shift=shift
    )

    # The shift should not affect the final matrix
    expected = np.eye(3)

    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize("translate, shear, scale, rotate, shift", [
    ({"x": 10, "y": 20}, {"x": 0, "y": 0}, {"x": 1, "y": 1}, 0, (0, 0)),
    ({"x": 0, "y": 0}, {"x": 30, "y": 0}, {"x": 2, "y": 2}, 45, (100, 100)),
    ({"x": -5, "y": 5}, {"x": 15, "y": 15}, {"x": 0.5, "y": 1.5}, 90, (-50, 50)),
])
def test_matrix_determinant(translate, shear, scale, rotate, shift):
    result = fgeometric.create_affine_transformation_matrix(translate, shear, scale, rotate, shift)
    det = np.linalg.det(result[:2, :2])
    expected_det_without_shear = scale["x"] * scale["y"]

    # Check if the determinant is positive
    assert det > 0, "Determinant should be positive"

    # Check if the determinant is real
    assert np.isclose(det, abs(det), rtol=1e-6), "Determinant should be real (close to its absolute value)"

    # If there's no shear, the determinant should be close to the product of scale factors
    if shear["x"] == 0 and shear["y"] == 0:
        assert np.isclose(det, expected_det_without_shear, rtol=1e-6), \
            f"Without shear, determinant {det} should be close to expected {expected_det_without_shear}"
    else:
        # With shear, we expect the determinant to be different from the product of scale factors
        # We can check if it's within a reasonable range, e.g., 50% to 150% of the expected value without shear
        assert 0.5 * expected_det_without_shear < det < 1.5 * expected_det_without_shear, \
            f"With shear, determinant {det} should be within 50-150% of {expected_det_without_shear}"


@pytest.mark.parametrize("image_shape", image_shapes)
@pytest.mark.parametrize(
    "translation,padding_value",
    [
        ((10, 0), 0),  # Right translation with zero padding
        ((-10, 0), 0),  # Left translation with zero padding
        ((0, 10), 0),  # Down translation with zero padding
        ((0, -10), 0),  # Up translation with zero padding
    ],
)
def test_edge_padding(image_shape, translation, padding_value):
    # Create an image filled with ones
    image = np.ones(image_shape, dtype=np.uint8) * 255

    # Define parameters for the affine transformation
    translate: fgeometric.TranslateDict = {"x": translation[0], "y": translation[1]}
    shear: fgeometric.ShearDict = {"x": 0, "y": 0}
    scale: fgeometric.ScaleDict = {"x": 1, "y": 1}
    rotate = 0
    shift = fgeometric.center(image_shape)  # Center of the image

    # Create the affine transformation matrix
    affine_matrix = fgeometric.create_affine_transformation_matrix(
        translate=translate,
        shear=shear,
        scale=scale,
        rotate=rotate,
        shift=shift
    )

    # Apply the transformation with specified padding
    warped_img = fgeometric.warp_affine(
        image,
        affine_matrix,
        cv2.INTER_LINEAR,
        padding_value,
        cv2.BORDER_CONSTANT,
        image.shape[:2],
    )

    # Check if the edge padding is correctly applied
    if translation[0] > 0:  # Right translation
        assert np.all(
            warped_img[:, :translation[0]] == padding_value
        ), "Edge padding failed: Incorrect padding on left edge."
    elif translation[0] < 0:  # Left translation
        assert np.all(
            warped_img[:, translation[0]:] == padding_value
        ), "Edge padding failed: Incorrect padding on right edge."
    if translation[1] > 0:  # Down translation
        assert np.all(
            warped_img[:translation[1], :] == padding_value
        ), "Edge padding failed: Incorrect padding on top edge."
    elif translation[1] < 0:  # Up translation
        assert np.all(
            warped_img[translation[1]:, :] == padding_value
        ), "Edge padding failed: Incorrect padding on bottom edge."

    # Check that the non-padded area still contains the original values
    if translation[0] > 0:
        assert np.all(warped_img[:, translation[0]:] == 255), "Non-padded area was altered."
    elif translation[0] < 0:
        assert np.all(warped_img[:, :translation[0]] == 255), "Non-padded area was altered."
    if translation[1] > 0:
        assert np.all(warped_img[translation[1]:, :] == 255), "Non-padded area was altered."
    elif translation[1] < 0:
        assert np.all(warped_img[:translation[1], :] == 255), "Non-padded area was altered."


def calculate_iou(img1, img2):
    intersection = np.logical_and(img1, img2)
    union = np.logical_or(img1, img2)
    return np.sum(intersection) / np.sum(union)


@pytest.mark.parametrize(
    "image_shape",
    [
        (100, 100),
        (100, 100, 1),
        (100, 100, 3),
    ],
)
@pytest.mark.parametrize("angle", [45,
                                   90, 180
                                   ])  # Rotation angles
@pytest.mark.parametrize("shape", ["circle", "rectangle"])
@pytest.mark.parametrize("scale", [1,
                                   0.8,
                                   1.2
                                   ])  # Scale factors
def test_inverse_angle_scale(image_shape, angle, shape, scale):
    image = np.zeros(image_shape, dtype=np.uint8)

    if shape == "rectangle":
        cv2.rectangle(image, (25, 25), (75, 75), 255, -1)  # Adjust for clarity
    elif shape == "circle":
        cv2.circle(image, (50, 50), 25, 255, -1)

    center = fgeometric.center(image_shape)

    # Create forward transformation matrix
    forward_matrix = fgeometric.create_affine_transformation_matrix(
        translate={"x": 0, "y": 0},
        shear={"x": 0, "y": 0},
        scale={"x": scale, "y": scale},
        rotate=angle,
        shift=center
    )

    # Create inverse transformation matrix
    inverse_matrix = fgeometric.create_affine_transformation_matrix(
        translate={"x": 0, "y": 0},
        shear={"x": 0, "y": 0},
        scale={"x": 1/scale, "y": 1/scale},
        rotate=-angle,
        shift=center
    )

    # Apply the forward transformation
    warped_img = fgeometric.warp_affine(
        image,
        forward_matrix,
        cv2.INTER_NEAREST,
        0,
        cv2.BORDER_CONSTANT,
        image_shape[:2],
    )
    restored_img = fgeometric.warp_affine(
        warped_img,
        inverse_matrix,
        cv2.INTER_NEAREST,
        0,
        cv2.BORDER_CONSTANT,
        image_shape[:2],
    )

    # Calculate IoU
    iou = calculate_iou(image > 0, restored_img > 0)
        # Calculate difference statistics
    diff = np.abs(image.astype(int) - restored_img.astype(int))

    mean_diff = np.mean(diff)
    num_diff_pixels = np.sum(diff > 0)

    # Assert that the differences are within acceptable limits
    assert iou > 0.97, f"IoU ({iou}) is too low"
    assert mean_diff < 2, f"Mean difference ({mean_diff}) is too high"
    assert num_diff_pixels / image.size < 0.1, f"Too many different pixels ({num_diff_pixels})"

@pytest.mark.parametrize(
    "img,expected",
    [
        (
            np.array(
                [
                    [0.01, 0.02, 0.03, 0.04],
                    [0.05, 0.06, 0.07, 0.08],
                    [0.09, 0.10, 0.11, 0.12],
                    [0.13, 0.14, 0.15, 0.16],
                ],
                dtype=np.float32,
            ),
                        np.array(
                [
                    [0.01, 0.02, 0.03, 0.04],
                    [0.05, 0.06, 0.07, 0.08],
                    [0.09, 0.10, 0.11, 0.12],
                    [0.13, 0.14, 0.15, 0.16],
                ],
                dtype=np.float32,
            ),
        ),
        (
            np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.uint8),
            np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.uint8),
        ),    ],
)  # Scale factors
def test_scale_with_warp_affine(img, expected):
    scale = 2

        # Create a centered scaling matrix
    transform = fgeometric.create_affine_transformation_matrix(
        translate={"x": 0, "y": 0},
        shear={"x": 0, "y": 0},
        scale={"x": scale, "y": scale},
        rotate=0,
        shift=(0, 0)
    )

    expected_shape = (int(img.shape[0] * scale), int(img.shape[1] * scale))

    # Apply scaling using warp_affine from the example provided
    scaled_img = fgeometric.warp_affine(
        img=img,
        matrix=transform,
        interpolation=cv2.INTER_NEAREST,
        cval=0,
        output_shape=expected_shape,
        mode=cv2.BORDER_CONSTANT,
    )

    assert scaled_img.shape == expected_shape, f"Expected shape {expected_shape}, got {scaled_img.shape}"

    # Downscale the result back to the original size for comparison
    downscaled_img = cv2.resize(scaled_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    np.testing.assert_array_almost_equal(downscaled_img, expected, decimal=5)


@pytest.mark.parametrize(
    "img,expected",
    [
        (
            np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.uint8),
            np.array([[4, 8, 12, 16], [3, 7, 11, 15], [2, 6, 10, 14], [1, 5, 9, 13]], dtype=np.uint8),
        ),
        (
            np.array(
                [
                    [0.01, 0.02, 0.03, 0.04],
                    [0.05, 0.06, 0.07, 0.08],
                    [0.09, 0.10, 0.11, 0.12],
                    [0.13, 0.14, 0.15, 0.16],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [0.04, 0.08, 0.12, 0.16],
                    [0.03, 0.07, 0.11, 0.15],
                    [0.02, 0.06, 0.10, 0.14],
                    [0.01, 0.05, 0.09, 0.13],
                ],
                dtype=np.float32,
            ),
        ),
    ],
)
def test_rotate_with_warp_affine(img, expected):
    angle = 90

    center = fgeometric.center(img.shape[:2])

    # Create transformation matrix
    transform = fgeometric.create_affine_transformation_matrix(
        translate={"x": 0, "y": 0},
        shear={"x": 0, "y": 0},
        scale={"x": 1, "y": 1},
        rotate=angle,
        shift=center
    )

    # Apply scaling using warp_affine from the example provided
    scaled_img = fgeometric.warp_affine(
        img=img,
        matrix=transform,  # Use the top 2 rows of the 3x3 matrix
        interpolation=cv2.INTER_LINEAR,
        cval=0,
        output_shape=img.shape[:2],
        mode=cv2.BORDER_CONSTANT,
    )

    np.testing.assert_array_equal(scaled_img, expected)
    # Additional test to ensure the transformation matrix is correct
    expected_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    np.testing.assert_array_almost_equal(transform[:2], expected_matrix, decimal=5)


@pytest.mark.parametrize(
    "img,expected,translate",
    [
        (
            np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.uint8),
            np.array([[0, 0, 1, 2], [0, 0, 5, 6], [0, 0, 9, 10], [0, 0, 13, 14]], dtype=np.uint8),
            (0, 2),
        ),
        (
            np.array(
                [
                    [0.01, 0.02, 0.03, 0.04],
                    [0.05, 0.06, 0.07, 0.08],
                    [0.09, 0.10, 0.11, 0.12],
                    [0.13, 0.14, 0.15, 0.16],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [0.00, 0.00, 0.01, 0.02],
                    [0.00, 0.00, 0.05, 0.06],
                    [0.00, 0.00, 0.09, 0.10],
                    [0.00, 0.00, 0.13, 0.14],
                ],
                dtype=np.float32,
            ),
            (0, 2),
        ),
        (
            np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.uint8),
            np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint8),
            (2, 0),
        ),
        (
            np.array(
                [
                    [0.01, 0.02, 0.03, 0.04],
                    [0.05, 0.06, 0.07, 0.08],
                    [0.09, 0.10, 0.11, 0.12],
                    [0.13, 0.14, 0.15, 0.16],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [0.00, 0.00, 0.00, 0.00],
                    [0.00, 0.00, 0.00, 0.00],
                    [0.01, 0.02, 0.03, 0.04],
                    [0.05, 0.06, 0.07, 0.08],
                ],
                dtype=np.float32,
            ),
            (2, 0),
        ),
    ],
)
def test_translate_with_warp_affine(img, expected, translate):
    transform = fgeometric.create_affine_transformation_matrix(
        translate={"x": translate[1], "y": translate[0]},  # Negate both x and y
        shear={"x": 0, "y": 0},
        scale={"x": 1, "y": 1},
        rotate=0,
        shift=(0, 0)
    )

    # Apply scaling using warp_affine from the example provided
    scaled_img = fgeometric.warp_affine(
        img=img,
        matrix=transform,  # Use the top 2 rows of the 3x3 matrix
        interpolation=cv2.INTER_LINEAR,
        cval=0,
        output_shape=img.shape[:2],
        mode=cv2.BORDER_CONSTANT,
    )

    np.testing.assert_array_equal(scaled_img, expected)


@pytest.mark.parametrize(
    "image_shape",
    [
        (100, 100),
        (100, 100, 1),
        (100, 100, 3),
    ],
)
@pytest.mark.parametrize("shape", ["circle", "rectangle"])
@pytest.mark.parametrize("shear", [0, 20, -20])  # Shear angles
def test_inverse_shear(image_shape, shear, shape):
    image = np.zeros(image_shape, dtype=np.uint8)

    if shape == "rectangle":
        cv2.rectangle(image, (25, 25), (75, 75), 255, -1)  # Adjust for clarity
    elif shape == "circle":
        cv2.circle(image, (50, 50), 25, 255, -1)

    center = fgeometric.center(image_shape[:2])

    # Create forward transformation matrix
    forward_matrix = fgeometric.create_affine_transformation_matrix(
        translate={"x": 0, "y": 0},
        shear={"x": shear, "y": 0},  # Apply shear only in x-direction
        scale={"x": 1, "y": 1},
        rotate=0,
        shift=center
    )

    # Create inverse transformation matrix
    inverse_matrix = fgeometric.create_affine_transformation_matrix(
        translate={"x": 0, "y": 0},
        shear={"x": -shear, "y": 0},  # Apply inverse shear
        scale={"x": 1, "y": 1},
        rotate=0,
        shift=center
    )

    assert forward_matrix.shape == (3, 3), "Forward matrix should be 3x3"
    assert inverse_matrix.shape == (3, 3), "Inverse matrix should be 3x3"

    # Apply the forward transformation
    warped_img = fgeometric.warp_affine(
        image,
        forward_matrix,  # Use only the top two rows
        cv2.INTER_NEAREST,
        0,
        cv2.BORDER_CONSTANT,
        image_shape[:2],
    )

    # Apply the inverse transformation
    restored_img = fgeometric.warp_affine(
        warped_img,
        inverse_matrix,  # Use only the top two rows
        cv2.INTER_NEAREST,
        0,
        cv2.BORDER_CONSTANT,
        image_shape[:2],
    )

    # Check if the restored image is close to the original
    np.testing.assert_allclose(image, restored_img, atol=1,
                               err_msg=f"Inverse transformation failed: The restored image is not close enough to the original. "
                               f"Shape: {shape}, Shear: {shear}, Max difference: {np.max(np.abs(image - restored_img))}")

    # Check if the forward transformation actually changed the image (except for shear=0)
    if shear != 0:
        assert not np.array_equal(image, warped_img), f"Forward transformation had no effect. Shape: {shape}, Shear: {shear}"

@pytest.mark.parametrize(
    "image_shape",
    [
        (100, 100),
        (100, 100, 1),
        (100, 100, 3),
    ],
)
@pytest.mark.parametrize("shape", ["circle", "rectangle"])
@pytest.mark.parametrize("translate", [(0, 0), (10, -10), (-10, 10)])  # Translation vectors
def test_inverse_translate(image_shape, translate, shape):
    image = np.zeros(image_shape, dtype=np.uint8)

    if shape == "rectangle":
        cv2.rectangle(image, (25, 25), (75, 75), 255, -1)  # Adjust for clarity
    elif shape == "circle":
        cv2.circle(image, (50, 50), 25, 255, -1)

    center = fgeometric.center(image_shape[:2])

    # Create forward transformation matrix
    forward_matrix = fgeometric.create_affine_transformation_matrix(
        translate={"x": translate[0], "y": translate[1]},
        shear={"x": 0, "y": 0},
        scale={"x": 1, "y": 1},
        rotate=0,
        shift=center
    )

    # Create inverse transformation matrix
    inverse_matrix = fgeometric.create_affine_transformation_matrix(
        translate={"x": -translate[0], "y": -translate[1]},
        shear={"x": 0, "y": 0},
        scale={"x": 1, "y": 1},
        rotate=0,
        shift=center
    )

    # Apply the forward transformation
    warped_img = fgeometric.warp_affine(
        image,
        forward_matrix,
        cv2.INTER_NEAREST,
        0,
        cv2.BORDER_CONSTANT,
        image_shape[:2],
    )

    # Apply the inverse transformation
    restored_img = fgeometric.warp_affine(
        warped_img,
        inverse_matrix,
        cv2.INTER_NEAREST,
        0,
        cv2.BORDER_CONSTANT,
        image_shape[:2],
    )

    # Check if the restored image is identical to the original
    np.testing.assert_allclose(image, restored_img, atol=1,
                               err_msg=f"Inverse transformation failed: The restored image is not identical to the original. "
                               f"Shape: {shape}, Translation: {translate}, "
                               f"Max difference: {np.max(np.abs(image - restored_img))}")


    # Check if the forward transformation actually changed the image (except for translate=(0,0))
    if translate != (0, 0):
        assert not np.array_equal(image, warped_img), (
            f"Forward transformation had no effect. "
            f"Shape: {shape}, Translation: {translate}"
        )


@pytest.mark.parametrize(
    ["keypoint", "expected", "angle", "scale", "dx", "dy"],
    [[[50, 50, 0, 5], [120.5, 158.5,  math.pi / 2, 10], 90, 2, 0.1, 0.1]],
)
def test_keypoint_affine(keypoint, expected, angle, scale, dx, dy):
    height, width = 100, 200
    center = fgeometric.center((height, width))

    # Create forward transformation matrix
    forward_matrix = fgeometric.create_affine_transformation_matrix(
        translate={"x": dx * width, "y": dy * height},
        shear={"x": 0, "y": 0},
        scale={"x": scale, "y": scale},
        rotate=angle,
        shift=center
    )

    keypoints = np.array([keypoint])

    actual = fgeometric.keypoints_affine(
        keypoints,
        forward_matrix,
        (height, width),
        {"x": scale, "y": scale},
        cv2.BORDER_CONSTANT,
    )

    np.testing.assert_allclose(actual[0], expected, rtol=1e-4), f"Expected: {expected}, Actual: {actual}"

    # Additional test to verify inverse transformation
    inverse_matrix = np.linalg.inv(forward_matrix)

    restored_keypoints = fgeometric.keypoints_affine(
        actual,
        inverse_matrix,
        (height, width),
        {"x": 1/scale, "y": 1/scale},
        cv2.BORDER_CONSTANT,
    )
    np.testing.assert_allclose(restored_keypoints[0], keypoint, atol=1e-6), (
        f"Inverse transformation failed. Original: {keypoint}, Restored: {restored_keypoints[0]}"
    )


@pytest.mark.parametrize(
    "params, expected_matrix",
    [
        # Identity transformation
        (
            {
                "translate": {"x": 0, "y": 0},
                "shear": {"x": 0, "y": 0},
                "scale": {"x": 1, "y": 1},
                "rotate": 0,
                "shift": (0, 0),
            },
            np.eye(3),
        ),
        # Translation
        (
            {
                "translate": {"x": 10, "y": 20},
                "shear": {"x": 0, "y": 0},
                "scale": {"x": 1, "y": 1},
                "rotate": 0,
                "shift": (0, 0),
            },
            np.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]]),
        ),
        # Scaling
        (
            {
                "translate": {"x": 0, "y": 0},
                "shear": {"x": 0, "y": 0},
                "scale": {"x": 2, "y": 3},
                "rotate": 0,
                "shift": (0, 0),
            },
            np.array([[2, 0, 0], [0, 3, 0], [0, 0, 1]]),
        ),
        # Rotation (45 degrees)
        (
            {
                "translate": {"x": 0, "y": 0},
                "shear": {"x": 0, "y": 0},
                "scale": {"x": 1, "y": 1},
                "rotate": 45,
                "shift": (0, 0),
            },
            np.array(
                [[np.cos(np.pi / 4), np.sin(np.pi / 4), 0], [-np.sin(np.pi / 4), np.cos(np.pi / 4), 0], [0, 0, 1]],
            ),
        ),
        # Shear in x-direction
        (
            {
                "translate": {"x": 0, "y": 0},
                "shear": {"x": 30, "y": 0},
                "scale": {"x": 1, "y": 1},
                "rotate": 0,
                "shift": (0, 0),
            },
            np.array([[1, np.tan(np.pi / 6), 0], [0, 1, 0], [0, 0, 1]]),
        ),
        # Shear in y-direction
        (
            {
                "translate": {"x": 0, "y": 0},
                "shear": {"x": 0, "y": 30},
                "scale": {"x": 1, "y": 1},
                "rotate": 0,
                "shift": (0, 0),
            },
            np.array([[1, 0, 0], [np.tan(np.pi / 6), 1, 0], [0, 0, 1]]),
        ),
        # Complex transformation
        (
            {
                "translate": {"x": 10, "y": 20},
                "shear": {"x": 15, "y": 10},
                "scale": {"x": 2, "y": 3},
                "rotate": 30,
                "shift": (100, 100),
            },
            None,  # We'll compute this separately due to its complexity
        ),
    ],
)
def test_create_affine_transformation_matrix(params, expected_matrix):
    result = fgeometric.create_affine_transformation_matrix(**params)

    if expected_matrix is not None:
        np.testing.assert_allclose(result, expected_matrix)
    else:
        # For complex transformation, we'll check properties instead of the exact matrix
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)

        # Check if the matrix is invertible (determinant is not zero)
        assert np.abs(np.linalg.det(result)) > 1e-6

        # Check if the last row is [0, 0, 1] (characteristic of affine transformations)
        np.testing.assert_allclose(result[2], [0, 0, 1], atol=1e-6)

        # Additional checks for the complex transformation
        # Check if translation is applied
        assert result[0, 2] != 0 or result[1, 2] != 0

        # Check if rotation or shear is applied (off-diagonal elements are non-zero)
        assert result[0, 1] != 0 or result[1, 0] != 0

        # Check if scaling is applied (diagonal elements are not 1)
        assert result[0, 0] != 1 or result[1, 1] != 1


@pytest.mark.parametrize("image_shape", [(100, 100), (200, 150)])
@pytest.mark.parametrize(
    "transform_params",
    [
        # No transformation
        {"translate": {"x": 0, "y": 0}, "shear": {"x": 0, "y": 0}, "scale": {"x": 1, "y": 1}, "rotate": 0},
        # Only rotation
        {"translate": {"x": 0, "y": 0}, "shear": {"x": 0, "y": 0}, "scale": {"x": 1, "y": 1}, "rotate": 45},
        # Only scaling
        {"translate": {"x": 0, "y": 0}, "shear": {"x": 0, "y": 0}, "scale": {"x": 1.5, "y": 1.2}, "rotate": 0},
        # # Only shearing
        {"translate": {"x": 0, "y": 0}, "shear": {"x": 30, "y": 20}, "scale": {"x": 1, "y": 1}, "rotate": 0},
        # Rotation and scaling
        {"translate": {"x": 0, "y": 0}, "shear": {"x": 0, "y": 0}, "scale": {"x": 1.5, "y": 1.2}, "rotate": 45},
        # # Rotation and shearing
        {"translate": {"x": 0, "y": 0}, "shear": {"x": 30, "y": 20}, "scale": {"x": 1, "y": 1}, "rotate": 45},
        # # Scaling and shearing
        {"translate": {"x": 0, "y": 0}, "shear": {"x": 30, "y": 20}, "scale": {"x": 1.5, "y": 1.2}, "rotate": 0},
        # # All transformations combined
        {"translate": {"x": 0, "y": 0}, "shear": {"x": 30, "y": 20}, "scale": {"x": 1.5, "y": 1.2}, "rotate": 45},
    ],
)
def test_center_remains_stationary(image_shape, transform_params):
    center = fgeometric.center_bbox(image_shape)

    transform = fgeometric.create_affine_transformation_matrix(**transform_params, shift=center)

    # Apply the transformation to the center point
    center_homogeneous = np.array([center[0], center[1], 1])
    transformed_center = np.dot(transform, center_homogeneous)
    transformed_center = transformed_center[:2] / transformed_center[2]  # Convert back to Cartesian coordinates

    # Check if the center point remains stationary (allowing for small numerical errors)
    np.testing.assert_allclose(transformed_center, center, atol=1e-6)

    # If there's a translation, check if the center has moved by the expected amount
    if transform_params['translate']['x'] != 0 or transform_params['translate']['y'] != 0:
        expected_center = [
            center[0] + transform_params['translate']['x'],
            center[1] + transform_params['translate']['y']
        ]
        np.testing.assert_allclose(transformed_center, expected_center, atol=1e-6)


@pytest.mark.parametrize(
    ["image_shape", "transform_params", "expected_padding"],
    [
        # Test case 1: No transformation (identity matrix)
        (
            (100, 100),
            {"translate": {"x": 0, "y": 0}, "shear": {"x": 0, "y": 0}, "scale": {"x": 1, "y": 1}, "rotate": 0},
            (0, 0, 0, 0),
        ),
        # Test case 2: 45-degree rotation
        (
            (100, 100),
            {"translate": {"x": 0, "y": 0}, "shear": {"x": 0, "y": 0}, "scale": {"x": 1, "y": 1}, "rotate": 45},
            (51, 51, 51, 51),
        ),
        # Test case 3: Scale up by 2x
        (
            (100, 100),
            {"translate": {"x": 0, "y": 0}, "shear": {"x": 0, "y": 0}, "scale": {"x": 2, "y": 2}, "rotate": 0},
            (0, 0, 0, 0),
        ),
        # Test case 4: Translation
        (
            (100, 100),
            {"translate": {"x": 20, "y": 0}, "shear": {"x": 0, "y": 0}, "scale": {"x": 1, "y": 1}, "rotate": 0},
            (20, 0, 0, 0),
        ),
        (
            (100, 100),
            {"translate": {"x": 0, "y": 30}, "shear": {"x": 0, "y": 0}, "scale": {"x": 1, "y": 1}, "rotate": 0},
            (0, 0, 30, 0),
        ),
        (
            (100, 100),
            {"translate": {"x": 20, "y": 30}, "shear": {"x": 0, "y": 0}, "scale": {"x": 1, "y": 1}, "rotate": 0},
            (20, 0, 30, 0),
        ),
        # Test case 5: Rotation and scale
        (
            (100, 100),
            {"translate": {"x": 0, "y": 0}, "shear": {"x": 0, "y": 0}, "scale": {"x": 1.5, "y": 1.5}, "rotate": 30},
            (44, 44, 44, 44),
        ),
        # Test case 6: Rotation, scale, and translation
        (
            (100, 100),
            {"translate": {"x": 10, "y": -10}, "shear": {"x": 0, "y": 0}, "scale": {"x": 1.2, "y": 1.2}, "rotate": 60},
            (44, 44, 44, 44),
        ),
        # Test case 7: Non-square image
        (
            (150, 100),
            {"translate": {"x": 0, "y": 0}, "shear": {"x": 0, "y": 0}, "scale": {"x": 1, "y": 1}, "rotate": 45},
            (76, 76, 51, 51),
        ),
        # Test case 8: Shear transformation
        (
            (100, 100),
            {"translate": {"x": 0, "y": 0}, "shear": {"x": 30, "y": 0}, "scale": {"x": 1, "y": 1}, "rotate": 0},
            (58, 58, 0, 0),
        ),
        (
            (100, 100),
            {"translate": {"x": 0, "y": 0}, "shear": {"x": 0, "y": 0}, "scale": {"x": 0.5, "y": 0.5}, "rotate": 0},
            (50, 50, 50, 50),
        ),
        # Test case 10: Scale down by 0.5x with 45-degree rotation
        (
            (100, 100),
            {"translate": {"x": 0, "y": 0}, "shear": {"x": 0, "y": 0}, "scale": {"x": 0.5, "y": 0.5}, "rotate": 45},
            (92, 92, 92, 92),
        ),
        # Test case 11: Scale down by 0.5x with translation
        (
            (100, 100),
            {"translate": {"x": 10, "y": 15}, "shear": {"x": 0, "y": 0}, "scale": {"x": 0.5, "y": 0.5}, "rotate": 0},
            (70, 30, 80, 20),
        ),
        # Test case 12: Scale down by 0.5x with shear
        (
            (100, 100),
            {"translate": {"x": 0, "y": 0}, "shear": {"x": 30, "y": 0}, "scale": {"x": 0.5, "y": 0.5}, "rotate": 0},
            (108, 108, 50, 50),
        ),
        # Test case 13: Scale down by 0.5x with rotation and translation
        (
            (100, 100),
            {"translate": {"x": 5, "y": -5}, "shear": {"x": 0, "y": 0}, "scale": {"x": 0.5, "y": 0.5}, "rotate": 30},
            (101, 73, 83, 91),
        ),
        # Test case 14: Non-square image scaled down by 0.5x with rotation
        (
            (150, 100),
            {"translate": {"x": 0, "y": 0}, "shear": {"x": 0, "y": 0}, "scale": {"x": 0.5, "y": 0.5}, "rotate": 45},
            (127, 127, 102, 102),
        ),
        (
            (150, 100),
            {"translate": {"x": 0, "y": 0}, "shear": {"x": 0, "y": 0}, "scale": {"x": 0.5, "y": 0.5}, "rotate": 0},
            (50, 50, 75, 75),
        ),
        # Test case 15: Complex transformation with scale down
        (
            (100, 100),
            {"translate": {"x": 10, "y": -10}, "shear": {"x": 15, "y": 5}, "scale": {"x": 0.5, "y": 0.7}, "rotate": 60},
            (144, 83, 39, 24),
        ),
    ],
)
def test_calculate_affine_transform_padding(image_shape, transform_params, expected_padding):
    bbox_shift = fgeometric.center_bbox(image_shape)

    transform = fgeometric.create_affine_transformation_matrix(**transform_params, shift=bbox_shift)

    padding = fgeometric.calculate_affine_transform_padding(transform, image_shape)
    np.testing.assert_allclose(padding, expected_padding, atol=1,
                               err_msg=f"Failed for params: {transform_params}")

@pytest.mark.parametrize("transform_params, image_shape, expected_padding, description", [
    (
        {"translate": {"x": -50, "y": -50}, "shear": {"x": 0, "y": 0}, "scale": {"x": 1.5, "y": 1.5}, "rotate": 60},
        (100, 100),
        None,  # We'll just check if it's non-negative
        "Rotation, scaling, and translation should have non-negative padding"
    ),
    (
        {"translate": {"x": 0, "y": 0}, "shear": {"x": 0, "y": 0}, "scale": {"x": 1, "y": 1}, "rotate": 0},
        (100, 100),
        (0, 0, 0, 0),
        "Identity transform should require no padding"
    ),
    (
        {"translate": {"x": 20, "y": 30}, "shear": {"x": 0, "y": 0}, "scale": {"x": 1, "y": 1}, "rotate": 0},
        (100, 100),
        (20, 0, 30, 0),
        "Translation should result in correct padding"
    ),
    (
        {"translate": {"x": 0, "y": 0}, "shear": {"x": 0, "y": 0}, "scale": {"x": 2, "y": 2}, "rotate": 0},
        (100, 100),
        (0, 0, 0, 0),
        "Scaling by 2 should result in 50 pixels padding on all sides"
    ),
    (
        {"translate": {"x": 0, "y": 0}, "shear": {"x": 0, "y": 0}, "scale": {"x": 0.5, "y": 0.5}, "rotate": 0},
        (100, 100),
        (50, 50, 50, 50),
        "Scaling by 2 should result in 50 pixels padding on all sides"
    ),

])
def test_calculate_affine_transform_padding_properties(transform_params, image_shape, expected_padding, description):
    center = fgeometric.center_bbox(image_shape)

    matrix = fgeometric.create_affine_transformation_matrix(**transform_params, shift=center)

    padding = fgeometric.calculate_affine_transform_padding(matrix, image_shape)

    if expected_padding is None:
        assert all(p >= 0 for p in padding), f"{description}: {padding}"
    else:
        assert padding == expected_padding, f"{description}: expected {expected_padding}, got {padding}"


@pytest.mark.parametrize(
    ["format", "bbox"],
    [("pascal_voc", (40, 40, 60, 60)),
     ("albumentations", (0.4, 0.4, 0.6, 0.6))
     ],
)
@pytest.mark.parametrize(
    ["transform_class", "transform_params"],
    [
        (A.Affine, {"rotate": (90, 90)}),
        (A.Rotate, {"limit": (90, 90)}),
        (A.RandomRotate90, {}),
    ],
)
def test_rotate_by_90_bboxes_symmetric_bbox(transform_class, transform_params, format, bbox):
    bbox_params = A.BboxParams(format=format, label_fields=["cl"])

    transform = A.Compose(
        [transform_class(p=1, **transform_params)],
        bbox_params=bbox_params,
    )

    img = np.ones((100, 100, 3)) * 255

    bbox_out1 = transform(image=img, bboxes=[bbox], cl=[0])["bboxes"][0]

    np.testing.assert_allclose(bbox_out1, bbox, atol=1e-6)


@pytest.mark.parametrize("scale, keep_ratio, balanced_scale, expected", [
    ({"x": 1, "y": 1}, False, False, {"x": 1, "y": 1}),
    ({"x": 1, "y": 1}, True, False, {"x": 1, "y": 1}),
    ({"x": (0.5, 1.5), "y": (0.5, 1.5)}, False, False, {"x": pytest.approx(1, abs=0.5), "y": pytest.approx(1, abs=0.5)}),
    ({"x": (0.5, 1.5), "y": (0.5, 1.5)}, True, False, lambda x: x["x"] == x["y"] and 0.5 <= x["x"] <= 1.5),
    ({"x": (0.5, 1.5), "y": (0.5, 1.5)}, False, True, lambda x: all(0.5 <= v <= 1.5 for v in x.values())),
    ({"x": (0.5, 2.0), "y": 1}, False, False, lambda x: 0.5 <= x["x"] <= 2.0 and x["y"] == 1),
    ({"x": 0.5, "y": (0.5, 1.5)}, True, False, lambda x: x["x"] == 0.5 and x["y"] == 0.5),
    ({"x": (0.8, 1.2), "y": (0.8, 1.2)}, False, True, lambda x: all(0.8 <= v <= 1.2 for v in x.values())),
])
def test_get_scale(scale, keep_ratio, balanced_scale, expected):
    result = A.Affine.get_scale(scale, keep_ratio, balanced_scale)

    if callable(expected):
        assert expected(result)
    else:
        assert result == expected

def test_get_scale_balanced_scale_behavior():
    scale = {"x": (0.5, 2.0), "y": (0.5, 2.0)}
    result = A.Affine.get_scale(scale, keep_ratio=False, balanced_scale=True)

    assert all(0.5 <= v <= 2.0 for v in result.values())
    assert any(v < 1.0 for v in result.values()) or any(v > 1.0 for v in result.values())

def test_get_scale_keep_ratio():
    scale = {"x": (0.5, 1.5), "y": (0.8, 1.2)}
    result = A.Affine.get_scale(scale, keep_ratio=True, balanced_scale=False)

    assert result["x"] == result["y"]
    assert 0.5 <= result["x"] <= 1.5


@pytest.mark.parametrize(
    "scale, keep_ratio, balanced_scale, expected_x_range, expected_y_range",
    [
        ({"x": (0.5, 2), "y": (0.5, 2)}, False, True, (0.5, 2), (0.5, 2)),
        ({"x": (1, 2), "y": (1, 2)}, True, True, (1, 2), (1, 2)),
        ({"x": (0.5, 1), "y": (0.5, 1)}, True, True, (0.5, 1), (0.5, 1)),
        ({"x": (0.5, 2), "y": (0.5, 2)}, False, False, (0.5, 2), (0.5, 2)),
        ({"x": (0.5, 2), "y": (0.5, 2)}, True, False, (0.5, 2), (0.5, 2)),
    ],
)
def test_get_random_scale(scale, keep_ratio, balanced_scale, expected_x_range, expected_y_range):
    result = A.Affine.get_scale(scale, keep_ratio, balanced_scale)

    assert expected_x_range[0] <= result["x"] <= expected_x_range[1], "x is out of range"

    if keep_ratio:
        assert result["y"] == result["x"], "y should be equal to x when keep_ratio is True"
    else:
        assert expected_y_range[0] <= result["y"] <= expected_y_range[1], "y is out of range"

    if balanced_scale:
        assert (
            expected_x_range[0] <= result["x"] < 1 or 1 < result["x"] <= expected_x_range[1]
        ), "x should be in the balanced range"
        assert (
            expected_y_range[0] <= result["y"] < 1 or 1 < result["y"] <= expected_x_range[1]
        ), "x should be in the balanced range"


@pytest.mark.parametrize("points, matrix, expected", [
    # Test case 1: Identity transformation
    (
        np.array([[1, 1], [2, 2], [3, 3]]),
        np.eye(3),
        np.array([[1, 1], [2, 2], [3, 3]])
    ),
    # Test case 2: Translation
    (
        np.array([[1, 1], [2, 2], [3, 3]]),
        np.array([[1, 0, 1], [0, 1, 2], [0, 0, 1]]),
        np.array([[2, 3], [3, 4], [4, 5]])
    ),
    # Test case 3: Scaling
    (
        np.array([[1, 1], [2, 2], [3, 3]]),
        np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]]),
        np.array([[2, 2], [4, 4], [6, 6]])
    ),
    # Test case 4: Rotation (90 degrees)
    (
        np.array([[1, 0], [0, 1], [1, 1]]),
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
        np.array([[0, 1], [-1, 0], [-1, 1]])
    ),
    # Test case 5: Shear
    (
        np.array([[1, 1], [2, 2], [3, 3]]),
        np.array([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[1.5, 1], [3, 2], [4.5, 3]])
    ),
    (
        np.array([]),
        np.array([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([])
    ),
])
def test_apply_affine_to_points(points, matrix, expected):
    result = fgeometric.apply_affine_to_points(points, matrix)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)

@pytest.mark.parametrize("points, matrix", [
    # Test case for invalid input shapes
    (np.array([[1, 1]]), np.eye(2)),  # Invalid matrix shape
    (np.array([1, 1]), np.eye(3)),    # Invalid points shape
])
def test_apply_affine_to_points_invalid_input(points, matrix):
    with pytest.raises((ValueError, IndexError)):
        fgeometric.apply_affine_to_points(points, matrix)


@pytest.mark.parametrize(
    "angle",
    [
        90,
        180,
        -90,
    ],
)
def test_rot90(bboxes, angle, keypoints):
    image = SQUARE_UINT8_IMAGE

    mask = image.copy()

    bboxes = np.array(bboxes, dtype=np.float32)
    keypoints = np.array(keypoints, dtype=np.float32)

    image_shape = image.shape[:2]
    normalized_bboxes = normalize_bboxes(bboxes, image_shape)

    angle2factor = {90: 1, 180: 2, -90: 3}

    transform = A.Compose(
        [A.Affine(rotate=(angle, angle), p=1)],
        bbox_params=A.BboxParams(format="pascal_voc"),
        keypoint_params=A.KeypointParams(format="xyas"),
    )

    transformed = transform(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)

    factor = angle2factor[angle]

    image_rotated = fgeometric.rot90(image, factor)
    mask_rotated = fgeometric.rot90(image, factor)
    bboxes_rotated = fgeometric.bboxes_rot90(normalized_bboxes, factor)
    bboxes_rotated = denormalize_bboxes(bboxes_rotated, image_shape)

    keypoints_rotated = fgeometric.keypoints_rot90(keypoints, factor, image_shape)

    np.testing.assert_array_equal(transformed["image"], image_rotated)
    np.testing.assert_array_equal(transformed["mask"], mask_rotated)

    np.testing.assert_array_almost_equal(transformed["bboxes"], bboxes_rotated, decimal=1e-5)
    # If we want to check all coordinates we need additionally to convert for keypoints angle to radians and back as Compose does it for us
    np.testing.assert_array_almost_equal(transformed["keypoints"][:, :2], keypoints_rotated[:, :2])
