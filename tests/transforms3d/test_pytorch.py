import numpy as np
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensor3D

def test_to_tensor_3d():
    transform = A.Compose([ToTensor3D(p=1)])

    # Create sample 3D data
    images = np.random.randint(0, 256, (64, 64, 64, 3), dtype=np.uint8)  # (D, H, W, C)
    masks = np.random.randint(0, 2, (64, 64, 64), dtype=np.uint8)  # (D, H, W)

    transformed = transform(
        images=images,
        masks=masks,
    )

    # Check outputs are torch.Tensor
    assert isinstance(transformed["images"], torch.Tensor)
    assert isinstance(transformed["masks"], torch.Tensor)

    # Check shapes
    assert transformed["images"].shape == (3, 64, 64, 64)  # (C, D, H, W)
    assert transformed["masks"].shape == (64, 64, 64)  # (D, H, W)

    # Test with transpose_mask=True and channeled mask
    transform = A.Compose([ToTensor3D(p=1, transpose_mask=True)])
    mask_with_channels = np.random.randint(0, 2, (64, 64, 64, 4), dtype=np.uint8)  # (D, H, W, C)

    transformed = transform(
        images=images,
        masks=mask_with_channels,
    )

    assert isinstance(transformed["images"], torch.Tensor)
    assert isinstance(transformed["masks"], torch.Tensor)
    assert transformed["masks"].shape == (4, 64, 64, 64)  # (C, D, H, W)
