import math
import numpy as np
import pytest
import skimage.transform
import cv2
import albumentations.augmentations.geometric.functional as fgeometric
import albumentations as A

from itertools import product

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

    aff_transform = skimage.transform.AffineTransform(rotation=np.deg2rad(angle), translation=translation, scale=(scale, scale), shear=np.deg2rad(shear))
    projective_transform = skimage.transform.ProjectiveTransform(matrix=aff_transform.params)
    warped_img = fgeometric.warp_affine(image, projective_transform, cv2.INTER_LINEAR, 0, cv2.BORDER_CONSTANT, output_shape)

    assert warped_img.shape[:2] == output_shape, "Output shape does not match the expected shape."

def create_test_image(shape):
    """
    Creates a test image with distinct values in each channel.
    """
    assert len(shape) == 3 and shape[2] >= 2
    image = np.zeros(shape, dtype=np.uint8)

    for channel in range(shape[2]):
        image[:, :, channel] = np.uint8(channel + 1)
    return image

@pytest.mark.parametrize("image_shape", image_shapes[2:])
def test_channel_integrity(image_shape):
    image = create_test_image(image_shape)

    # Define an affine transform that does not alter the image (identity transform)
    aff_transform = skimage.transform.AffineTransform(scale=(1, 1), rotation=0, translation=(0, 0))
    projective_transform = skimage.transform.ProjectiveTransform(matrix=aff_transform.params)

    # Apply the transformation
    warped_img = fgeometric.warp_affine(image, projective_transform, cv2.INTER_LINEAR, 0, cv2.BORDER_CONSTANT, image.shape[:2])

    # Verify that the channels remain unchanged
    assert np.array_equal(warped_img, image), "Channel integrity failed: Channels were altered by transformation."


@pytest.mark.parametrize("image_shape", image_shapes)
@pytest.mark.parametrize("translation,padding_value", [
    ((10, 0), 0),  # Right translation with zero padding
    ((-10, 0), 0),  # Left translation with zero padding
    ((0, 10), 0),  # Down translation with zero padding
    ((0, -10), 0),  # Up translation with zero padding
])
def test_edge_padding(image_shape, translation, padding_value):
    # Create an image filled with ones

    image = np.ones(image_shape, dtype=np.uint8) * 255

    # Define an affine transform for the translation
    aff_transform = skimage.transform.AffineTransform(translation=translation)
    projective_transform = skimage.transform.ProjectiveTransform(matrix=aff_transform.params)

    # Apply the transformation with specified padding
    warped_img = fgeometric.warp_affine(image, projective_transform, cv2.INTER_LINEAR, padding_value, cv2.BORDER_CONSTANT, image.shape[:2])

    # Check if the edge padding is correctly applied
    if translation[0] > 0:  # Right translation
        assert np.all(warped_img[:, :translation[0]] == padding_value), "Edge padding failed: Incorrect padding on left edge."
    elif translation[0] < 0:  # Left translation
        assert np.all(warped_img[:, translation[0]:] == padding_value), "Edge padding failed: Incorrect padding on right edge."
    if translation[1] > 0:  # Down translation
        assert np.all(warped_img[:translation[1], :] == padding_value), "Edge padding failed: Incorrect padding on top edge."
    elif translation[1] < 0:  # Up translation
        assert np.all(warped_img[translation[1]:, :] == padding_value), "Edge padding failed: Incorrect padding on bottom edge."

def create_centered_comprehensive_transform(image_shape, angle, shear, translate, scale):
    height, width = image_shape

    center_y, center_x = (height - 1) / 2, (width - 1) / 2

    # Combined transformations
    translate_to_origin = skimage.transform.SimilarityTransform(translation=(-center_x, -center_y))
    comprehensive_transform = skimage.transform.AffineTransform(
        rotation=np.deg2rad(angle), shear=np.deg2rad(shear),
        translation=translate, scale=(scale, scale))
    translate_back = skimage.transform.SimilarityTransform(translation=(center_x, center_y))

    forward_transform = translate_to_origin + comprehensive_transform + translate_back

    # Create inverse of the comprehensive transform
    inverse_comprehensive_transform = skimage.transform.AffineTransform(
        rotation=np.deg2rad(-angle), shear=np.deg2rad(-shear),
        translation=(-np.array(translate) / scale), scale=(1/scale, 1/scale))

    inverse_transform = translate_to_origin + inverse_comprehensive_transform + translate_back

    return forward_transform, inverse_transform


@pytest.mark.parametrize("image_shape", [
    (100, 100),
    (100, 100, 1),
    (100, 100, 3),
])
@pytest.mark.parametrize("angle", [45, 90, 180])  # Rotation angles
@pytest.mark.parametrize("shape", ["circle", "rectangle"])
@pytest.mark.parametrize("scale", [1, 0.8, 1.2])  # Scale factors
def test_inverse_angle_scale(image_shape, angle, shape, scale):
    image = np.zeros(image_shape, dtype=np.uint8)

    if shape == "rectangle":
        cv2.rectangle(image, (25, 25), (75, 75), 255, -1)  # Adjust for clarity
    elif shape == "circle":
        cv2.circle(image, (50, 50), 25, 255, -1)

    # Adjust the function to handle shearing, translation, and scaling as well
    centered_transform, inverse_transform = create_centered_comprehensive_transform(image_shape[:2], angle, 0, (0, 0), scale)

    forward_projective_transform = skimage.transform.ProjectiveTransform(matrix=centered_transform.params)
    inverse_projective_transform = skimage.transform.ProjectiveTransform(matrix=inverse_transform.params)

    # Apply the forward transformation
    warped_img = fgeometric.warp_affine(image, forward_projective_transform, cv2.INTER_LINEAR, 0, cv2.BORDER_CONSTANT, image_shape[:2])

    # Apply the inverse transformation
    restored_img = fgeometric.warp_affine(warped_img, inverse_projective_transform, cv2.INTER_LINEAR, 0, cv2.BORDER_CONSTANT, image_shape[:2])

    # Adjust the assertion threshold based on expected discrepancies from complex transformations
    assert np.mean(np.abs(image - restored_img)) < 7, "Inverse transformation failed: The restored image is not close enough to the original."


@pytest.mark.parametrize("img,expected", [ (np.array( [[0.01, 0.02, 0.03, 0.04], [0.05, 0.06, 0.07, 0.08], [0.09, 0.10, 0.11, 0.12], [0.13, 0.14, 0.15, 0.16]], dtype=np.float32, ), np.array( [[0.04, 0.08, 0.12, 0.16], [0.03, 0.07, 0.11, 0.15], [0.02, 0.06, 0.10, 0.14], [0.01, 0.05, 0.09, 0.13]], dtype=np.float32, ) ), ( np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.uint8), np.array([[6, 6, 7, 7], [6, 6, 7, 7], [10, 10, 11, 11], [10, 10, 11, 11]], dtype=np.uint8) )])  # Scale factors
def test_scale_with_warp_affine(img, expected):
    scale = 2  # Example scaling factor

    transform, _ = create_centered_comprehensive_transform(img.shape[:2], 0, 0, (0, 0), scale)

    expected_shape = (int(img.shape[0] * scale), int(img.shape[1] * scale))

    # Apply scaling using warp_affine from the example provided
    scaled_img = fgeometric.warp_affine(
        img=img,
        matrix=transform,  # Use the top 2 rows of the 3x3 matrix
        interpolation=cv2.INTER_LINEAR,
        cval=0,
        output_shape=expected_shape,
        mode=cv2.BORDER_CONSTANT,
    )

    assert scaled_img.shape == expected_shape, f"Expected shape {expected_shape}, got {scaled_img.shape}"
    np.array_equal(scaled_img, expected)

@pytest.mark.parametrize("img,expected",
                        [
                            (
                                 np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.uint8),
    np.array([[4, 8, 12, 16], [3, 7, 11, 15], [2, 6, 10, 14], [1, 5, 9, 13]], dtype=np.uint8)
                                ), (
np.array(
        [[0.01, 0.02, 0.03, 0.04], [0.05, 0.06, 0.07, 0.08], [0.09, 0.10, 0.11, 0.12], [0.13, 0.14, 0.15, 0.16]],
        dtype=np.float32,
    ),
    np.array(
        [[0.04, 0.08, 0.12, 0.16], [0.03, 0.07, 0.11, 0.15], [0.02, 0.06, 0.10, 0.14], [0.01, 0.05, 0.09, 0.13]],
        dtype=np.float32,
    )
                                )
                        ] )
def test_rotate_with_warp_affine(img, expected):
    angle = 90

    transform, _ = create_centered_comprehensive_transform(img.shape[:2], angle, 0, (0, 0), 1)

    # Apply scaling using warp_affine from the example provided
    scaled_img = fgeometric.warp_affine(
        img=img,
        matrix=transform,  # Use the top 2 rows of the 3x3 matrix
        interpolation=cv2.INTER_LINEAR,
        cval=0,
        output_shape=img.shape[:2],
        mode=cv2.BORDER_CONSTANT,
    )

    np.array_equal(scaled_img, expected)


@pytest.mark.parametrize("img,expected,translate",
                        [
                            ( np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.uint8), np.array([[0, 0, 1, 2], [0, 0, 5, 6], [0, 0, 9, 10], [0, 0, 13, 14]], dtype=np.uint8), (0, 2)),
( np.array( [[0.01, 0.02, 0.03, 0.04], [0.05, 0.06, 0.07, 0.08], [0.09, 0.10, 0.11, 0.12], [0.13, 0.14, 0.15, 0.16]], dtype=np.float32, ), np.array( [[0.00, 0.00, 0.01, 0.02], [0.00, 0.00, 0.05, 0.06], [0.00, 0.00, 0.09, 0.10], [0.00, 0.00, 0.13, 0.14]], dtype=np.float32, ), (0, 2) ),
( np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.uint8), np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint8), (2, 0) ),
( np.array( [[0.01, 0.02, 0.03, 0.04], [0.05, 0.06, 0.07, 0.08], [0.09, 0.10, 0.11, 0.12], [0.13, 0.14, 0.15, 0.16]], dtype=np.float32, ), np.array( [[0.00, 0.00, 0.00, 0.00], [0.00, 0.00, 0.00, 0.00], [0.01, 0.02, 0.03, 0.04], [0.05, 0.06, 0.07, 0.08]], dtype=np.float32, ), (2, 0)) ] )
def test_translate_with_warp_affine(img, expected, translate):
    transform, _ = create_centered_comprehensive_transform(img.shape[:2], 0, 0, translate, 1)

    # Apply scaling using warp_affine from the example provided
    scaled_img = fgeometric.warp_affine(
        img=img,
        matrix=transform,  # Use the top 2 rows of the 3x3 matrix
        interpolation=cv2.INTER_LINEAR,
        cval=0,
        output_shape=img.shape[:2],
        mode=cv2.BORDER_CONSTANT,
    )

    np.array_equal(scaled_img, expected)


@pytest.mark.parametrize("image_shape", [
    (100, 100),
    (100, 100, 1),
    (100, 100, 3),
])
@pytest.mark.parametrize("shape", ["circle", "rectangle"])
@pytest.mark.parametrize("shear", [0, 20, -20])  # Shear angles
def test_inverse_shear(image_shape, shear, shape):
    image = np.zeros(image_shape, dtype=np.uint8)

    if shape == "rectangle":
        cv2.rectangle(image, (25, 25), (75, 75), 255, -1)  # Adjust for clarity
    elif shape == "circle":
        cv2.circle(image, (50, 50), 25, 255, -1)

    # Adjust the function to handle shearing, translation, and scaling as well
    centered_transform, inverse_transform = create_centered_comprehensive_transform(image_shape[:2], 0, shear, (0, 0), 1)

    forward_projective_transform = skimage.transform.ProjectiveTransform(matrix=centered_transform.params)
    inverse_projective_transform = skimage.transform.ProjectiveTransform(matrix=inverse_transform.params)

    # Apply the forward transformation
    warped_img = fgeometric.warp_affine(image, forward_projective_transform, cv2.INTER_LINEAR, 0, cv2.BORDER_CONSTANT, image_shape[:2])

    # Apply the inverse transformation
    restored_img = fgeometric.warp_affine(warped_img, inverse_projective_transform, cv2.INTER_LINEAR, 0, cv2.BORDER_CONSTANT, image_shape[:2])

    # Adjust the assertion threshold based on expected discrepancies from complex transformations
    assert np.array_equal(image - restored_img), "Inverse transformation failed: The restored image is not close enough to the original."


@pytest.mark.parametrize("image_shape", [
    (100, 100),
    (100, 100, 1),
    (100, 100, 3),
])

@pytest.mark.parametrize("shape", ["circle", "rectangle"])
@pytest.mark.parametrize("translate", [(0, 0), (10, -10), (-10, 10)])  # Translation vectors
def test_inverse_shear(image_shape, translate, shape):
    image = np.zeros(image_shape, dtype=np.uint8)

    if shape == "rectangle":
        cv2.rectangle(image, (25, 25), (75, 75), 255, -1)  # Adjust for clarity
    elif shape == "circle":
        cv2.circle(image, (50, 50), 25, 255, -1)

    # Adjust the function to handle shearing, translation, and scaling as well
    centered_transform, inverse_transform = create_centered_comprehensive_transform(image_shape[:2], 0, 0, translate, 1)

    forward_projective_transform = skimage.transform.ProjectiveTransform(matrix=centered_transform.params)
    inverse_projective_transform = skimage.transform.ProjectiveTransform(matrix=inverse_transform.params)

    # Apply the forward transformation
    warped_img = fgeometric.warp_affine(image, forward_projective_transform, cv2.INTER_LINEAR, 0, cv2.BORDER_CONSTANT, image_shape[:2])

    # Apply the inverse transformation
    restored_img = fgeometric.warp_affine(warped_img, inverse_projective_transform, cv2.INTER_LINEAR, 0, cv2.BORDER_CONSTANT, image_shape[:2])

    # Adjust the assertion threshold based on expected discrepancies from complex transformations
    assert np.array_equal(image, restored_img), "Inverse transformation failed: The restored image is not close enough to the original."


@pytest.mark.parametrize(
    ["keypoint", "expected", "angle", "scale", "dx", "dy"],
    [[[50, 50, 0, 5], [118, -40, math.pi / 2, 10], 90, 2, 0.1, 0.1]],
)
def test_keypoint_affine(keypoint, expected, angle, scale, dx, dy):
    height, width = 100, 200

    centered_transform, _ = create_centered_comprehensive_transform((height, width), angle, 0, (dx * width, dy * height), scale)

    transform = skimage.transform.ProjectiveTransform(matrix=centered_transform.params)

    actual = fgeometric.keypoint_affine(keypoint, transform, {"x": keypoint[2], "y": keypoint[3]})
    np.testing.assert_allclose(actual[:2], expected[:2], rtol=1e-4)


def test_affine_with_fit_output():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = [10, 10, 90, 90]  # [x_min, y_min, x_max, y_max]

    transform = A.Affine(rotate=45, scale=1.5, fit_output=True, p=1.0)
    transformed = transform(image=image, bboxes=[bbox])

    assert transformed['image'].shape != image.shape
    assert transformed['bboxes'][0] != bbox


def test_affine_translation_with_fit_output():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = [25, 25, 75, 75]

    transform = A.Affine(translate_percent={'x': 0.5, 'y': 0.5}, fit_output=True, p=1.0)
    transformed = transform(image=image, bboxes=[bbox])

    # Check if bbox has moved in the correct direction
    t_bbox = transformed['bboxes'][0]
    assert t_bbox[0] >= bbox[0] and t_bbox[1] >= bbox[1]
