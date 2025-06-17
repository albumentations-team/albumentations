"""Tests for worker-aware seed functionality in Compose."""

import multiprocessing
import sys

import numpy as np
import pytest

import albumentations as A


class MockWorkerInfo:
    """Mock torch.utils.data.get_worker_info() response."""
    def __init__(self, id: int):
        self.id = id


def test_worker_seed_without_torch():
    """Test that worker seed functionality works when PyTorch is not available."""
    # Create compose (worker-aware seed is now always enabled)
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
    ], seed=137)

    # Should work fine without PyTorch
    img = np.ones((100, 100, 3), dtype=np.uint8)
    result = transform(image=img)
    assert result['image'].shape == (100, 100, 3)


@pytest.mark.skipif(
    "torch" not in sys.modules and not any("torch" in str(p) for p in sys.path),
    reason="PyTorch not available"
)
def test_worker_seed_with_torch():
    """Test worker seed functionality with PyTorch available."""
    import torch
    import torch.utils.data

    class TestDataset(torch.utils.data.Dataset):
        def __init__(self, transform):
            self.transform = transform

        def __len__(self):
            return 10

        def __getitem__(self, idx):
            # Create a simple test image
            img = np.ones((100, 100, 3), dtype=np.uint8) * idx
            result = self.transform(image=img)
            # Return whether the image was flipped
            return np.array_equal(result['image'], img)

    # Test with worker-aware seed (now always enabled)
    transform_with_worker_seed = A.Compose([
        A.HorizontalFlip(p=0.5),
    ], seed=137)

    dataset = TestDataset(transform_with_worker_seed)

    # Test with multiple workers
    if sys.platform != "darwin" or sys.version_info < (3, 8):
        # Skip multiprocessing tests on macOS with older Python due to fork/spawn issues
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=2,
            persistent_workers=False
        )

        # Collect results from multiple epochs
        epoch_results = []
        for epoch in range(3):
            epoch_data = []
            for batch in loader:
                epoch_data.append(batch.item())
            epoch_results.append(epoch_data)

        # With worker-aware seed and persistent_workers=False,
        # results should vary between epochs
        assert epoch_results[0] != epoch_results[1] or epoch_results[1] != epoch_results[2], \
            "All epochs produced identical results with worker-aware seed"


@pytest.mark.skipif(
    "torch" not in sys.modules and not any("torch" in str(p) for p in sys.path),
    reason="PyTorch not available"
)
def test_dataloader_epoch_diversity():
    """Test that DataLoader produces different augmentations across epochs with worker-aware seed."""
    import torch
    import torch.utils.data

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, transform):
            self.transform = transform
            # Create identical images
            self.data = [np.ones((10, 10, 3), dtype=np.uint8) * 255] * 4

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            image = self.data[idx].copy()
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            # Return sum of pixel values as a simple hash
            return float(np.sum(image))

    # Create transform with fixed seed (worker-aware seed is always enabled)
    transform = A.Compose([
        A.RandomBrightnessContrast(p=1.0, brightness_limit=0.3),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
    ], seed=42)

    dataset = SimpleDataset(transform=transform)

    # Skip on platforms where multiprocessing is problematic
    if sys.platform == "darwin" and sys.version_info >= (3, 8):
        pytest.skip("Multiprocessing test skipped on macOS with Python 3.8+")

    # Test with persistent_workers=False
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        persistent_workers=False
    )

    # Collect data from multiple epochs
    epoch_data = []
    for epoch in range(3):
        epoch_batch_sums = []
        for batch in dataloader:
            # Convert batch to list and sum all values
            batch_sum = float(torch.sum(batch))
            epoch_batch_sums.append(batch_sum)
        epoch_data.append(epoch_batch_sums)

    # Check that epochs produce different results
    # At least one epoch should differ from the others
    assert not (epoch_data[0] == epoch_data[1] == epoch_data[2]), \
        f"All epochs produced identical augmentations: {epoch_data}"


def test_compose_serialization():
    """Test that Compose serialization works properly."""
    # Test with worker-aware seed (always enabled)
    transform1 = A.Compose([
        A.HorizontalFlip(p=0.5),
    ], seed=137)

    # Serialize and deserialize
    serialized = transform1.to_dict()

    # Test deserialization
    transform2 = A.from_dict(serialized)
    assert hasattr(transform2, 'seed')
    assert transform2.seed == 137


def test_effective_seed_calculation():
    """Test the _get_effective_seed method directly."""
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
    ], seed=137)

    # Test with None seed
    assert transform._get_effective_seed(None) is None

    # Test without worker context
    assert transform._get_effective_seed(137) == 137

    # Test seed overflow
    large_seed = 2**32 - 1
    result = transform._get_effective_seed(large_seed)
    assert 0 <= result < 2**32


def test_deterministic_behavior_single_process():
    """Test that transforms are deterministic in a single process."""
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
    ], seed=137)

    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    # Reset seed and get results
    results = []
    for _ in range(3):
        transform.set_random_seed(137)
        result = transform(image=img.copy())
        results.append(result['image'])

    # All results should be identical
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])


def test_multiple_compose_instances():
    """Test that multiple Compose instances with same seed produce same results."""
    # Create two instances with same configuration
    transform1 = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5),
    ], seed=137)

    transform2 = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5),
    ], seed=137)

    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    # Both should produce the same result
    result1 = transform1(image=img.copy())
    result2 = transform2(image=img.copy())

    np.testing.assert_array_equal(result1['image'], result2['image'])


def worker_process_simulation(worker_id: int, base_seed: int, num_iterations: int) -> list[bool]:
    """Simulate a worker process with given ID and seed.

    Returns list of booleans indicating whether HorizontalFlip was applied.
    """
    # Each worker uses a different seed to simulate worker diversity
    # This simulates what would happen with torch.initial_seed()
    # Use a hash to get more diverse seeds
    import hashlib
    worker_seed = int(hashlib.md5(f"{base_seed}_{worker_id}".encode()).hexdigest()[:8], 16)

    # Create transform with a unique seed per worker
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
    ], seed=worker_seed)  # Simulating worker-aware behavior

    # Run iterations
    results = []
    # Create an asymmetric image so we can detect flips
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :5] = 255  # Left half white

    for _ in range(num_iterations):
        result = transform(image=img.copy())
        # Check if image was flipped by checking left corner
        was_flipped = result['image'][0, 0, 0] == 0  # If flipped, left corner will be black
        results.append(was_flipped)

    return results


def test_worker_seed_diversity():
    """Test that different workers produce different augmentation sequences."""
    if sys.platform == "win32":
        # Skip on Windows due to multiprocessing limitations
        pytest.skip("Multiprocessing test skipped on Windows")

    base_seed = 137
    num_workers = 4
    num_iterations = 20

    # Run simulation for each worker
    with multiprocessing.Pool(processes=num_workers) as pool:
        worker_results = []
        for worker_id in range(num_workers):
            result = pool.apply_async(
                worker_process_simulation,
                args=(worker_id, base_seed, num_iterations)
            )
            worker_results.append(result)

        # Collect results
        sequences = [result.get() for result in worker_results]

    # Check that workers produced different sequences
    unique_sequences = {tuple(seq) for seq in sequences}
    assert len(unique_sequences) > 1, "All workers produced identical augmentation sequences"

    # Each worker should have some flips and some non-flips (with high probability)
    for worker_id, sequence in enumerate(sequences):
        num_flips = sum(sequence)
        assert 0 < num_flips < num_iterations, \
            f"Worker {worker_id} produced extreme results: {num_flips}/{num_iterations} flips"
