import pytest
import numpy as np
import torch
import albumentations as A

test_cases = [
    pytest.param(
        (64, 64, 64, 3),  # volume shape
        (64, 64, 64, 1),  # mask shape
        (3, 64, 64, 64),  # expected volume shape
        (1, 64, 64, 64),  # expected mask shape
        id="rgb_volume_single_channel_mask"
    ),
    pytest.param(
        (64, 64, 64),     # volume shape
        (64, 64, 64),     # mask shape
        (1, 64, 64, 64),  # expected volume shape
        (1, 64, 64, 64),  # expected mask shape
        id="grayscale_volume_and_mask"
    ),
    pytest.param(
        (64, 64, 64, 3),  # volume shape
        (64, 64, 64, 4),  # mask shape
        (3, 64, 64, 64),  # expected volume shape
        (4, 64, 64, 64),  # expected mask shape
        id="rgb_volume_multi_channel_mask"
    ),
]

@pytest.mark.parametrize(
    "volume_shape,mask_shape,expected_volume_shape,expected_mask_shape",
    test_cases
)
def test_to_tensor_3d_shapes(
    volume_shape,
    mask_shape,
    expected_volume_shape,
    expected_mask_shape
):
    transform = A.Compose([A.ToTensor3D(p=1)])
    volume = np.random.randint(0, 256, volume_shape, dtype=np.uint8)
    mask3d = np.random.randint(0, 2, mask_shape, dtype=np.uint8)

    transformed = transform(volume=volume, mask3d=mask3d)

    assert isinstance(transformed["volume"], torch.Tensor)
    assert isinstance(transformed["mask3d"], torch.Tensor)
    assert transformed["volume"].shape == expected_volume_shape
    assert transformed["mask3d"].shape == expected_mask_shape


error_test_cases = [
    pytest.param(
        (64, 64),        # 2D array
        TypeError,
        "volume must be 3D or 4D array",
        id="2d_array"
    ),
    pytest.param(
        (64, 64, 64, 3, 1),  # 5D array
        TypeError,
        "volume must be 3D or 4D array",
        id="5d_array"
    ),
]

@pytest.mark.parametrize(
    "volume_shape,expected_error,expected_message",
    error_test_cases
)
def test_to_tensor_3d_errors(volume_shape, expected_error, expected_message):
    transform = A.Compose([A.ToTensor3D(p=1)])
    volume = np.random.rand(*volume_shape)

    with pytest.raises(expected_error, match=expected_message):
        transform(volume=volume)
