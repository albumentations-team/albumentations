import pytest
import numpy as np
import cv2
from albumentations.augmentations.domain_adaptation_functional import MinMaxScaler, StandardScaler, PCA, apply_histogram


@pytest.mark.parametrize("feature_range, data, expected", [
    ((0.0, 1.0), np.array([[1, 2], [3, 4], [5, 6]]), np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])),
    ((-1.0, 1.0), np.array([[1, 2], [3, 4], [5, 6]]), np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])),
    ((0.0, 1.0), np.array([[1, 1], [1, 1], [1, 1]]), np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])),  # edge case: all values are the same
])
def test_minmax_scaler(feature_range, data, expected):
    scaler = MinMaxScaler(feature_range)
    result = scaler.fit_transform(data)
    np.testing.assert_almost_equal(result, expected)

@pytest.mark.parametrize("feature_range, data, data_scaled, expected", [
    ((0.0, 1.0), np.array([[1, 2], [3, 4], [5, 6]]), np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]), np.array([[1, 2], [3, 4], [5, 6]])),
    ((-1.0, 1.0), np.array([[1, 2], [3, 4], [5, 6]]), np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]), np.array([[1, 2], [3, 4], [5, 6]])),
])
def test_minmax_inverse_transform(feature_range, data, data_scaled, expected):
    scaler = MinMaxScaler(feature_range)
    scaler.fit(data)
    result = scaler.inverse_transform(data_scaled)
    np.testing.assert_almost_equal(result, expected)

@pytest.mark.parametrize("data, expected", [
    (np.array([[1, 2], [3, 4], [5, 6]]), np.array([[-1.22474487, -1.22474487], [0.0, 0.0], [1.22474487, 1.22474487]])),
    (np.array([[1, 1], [1, 1], [1, 1]]), np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])),  # edge case: all values are the same
])
def test_standard_scaler(data, expected):
    scaler = StandardScaler()
    result = scaler.fit_transform(data)
    np.testing.assert_almost_equal(result, expected)

@pytest.mark.parametrize("data, data_scaled, expected", [
    (np.array([[1, 2], [3, 4], [5, 6]]), np.array([[-1.22474487, -1.22474487], [0.0, 0.0], [1.22474487, 1.22474487]]), np.array([[1, 2], [3, 4], [5, 6]])),
])
def test_standard_inverse_transform(data, data_scaled, expected):
    scaler = StandardScaler()
    scaler.fit(data)
    result = scaler.inverse_transform(data_scaled)
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize("n_components, data, expected_shape", [
    (1, np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0],
                  [2.3, 2.7], [2.0, 1.6], [1.0, 1.1], [1.5, 1.6], [1.1, 0.9]]), (10, 1)),
    (2, np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0],
                  [2.3, 2.7], [2.0, 1.6], [1.0, 1.1], [1.5, 1.6], [1.1, 0.9]]), (10, 2)),
])
def test_pca_transform(n_components, data, expected_shape):
    pca = PCA(n_components)
    transformed = pca.fit_transform(data)
    assert transformed.shape == expected_shape
    assert np.all(np.isfinite(transformed)), "Transformed data contains non-finite values"


@pytest.mark.parametrize("n_components, data, expected_transformed, expected_inverse",
    [(1, np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0],
                  [2.3, 2.7], [2.0, 1.6], [1.0, 1.1], [1.5, 1.6], [1.1, 0.9]]),
    np.array([[0.8279701862010879], [-1.7775803252804292], [0.9921974944148886], [0.27421041597539936], [1.6758014186445398], [0.9129491031588081], [-0.09910943749844434], [-1.1445721637986601], [-0.4380461367624502], [-1.2238205550547405]]),
    np.array([[2.3712589640000026, 2.518706008322169], [0.6050255837456271, 0.6031608863381424], [2.4825842875499986, 2.639442419978468], [1.995879946589024, 2.111593644953067], [2.9459812029146377, 3.142013433918504], [2.428863911241362, 2.5811806942407656], [1.7428163487767303, 1.837136856988131], [1.0341249774652423, 1.068534975444947], [1.5130601765607719, 1.58795783010856], [0.9804046011566055, 1.0102732497072444]])

     ),
    (2, np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0],
                  [2.3, 2.7], [2.0, 1.6], [1.0, 1.1], [1.5, 1.6], [1.1, 0.9]]),
     np.array([[0.8279701862010879, 0.17511530704691558], [-1.7775803252804292, -0.14285722654428068], [0.9921974944148886, -0.3843749888804125], [0.27421041597539936, -0.13041720657412711], [1.6758014186445398, 0.20949846125675342], [0.9129491031588081, -0.17528244362036988], [-0.09910943749844434, 0.34982469809712086], [-1.1445721637986601, -0.04641725818328135], [-0.4380461367624502, -0.017764629675083188], [-1.2238205550547405, 0.16267528707676182]]),
     np.array([[2.5, 2.4], [0.4999999999999998, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1000000000000005, 3.0], [2.3, 2.7], [2.0, 1.6], [0.9999999999999999, 1.1], [1.5, 1.6], [1.0999999999999999, 0.9000000000000001]]),
     ),
])
def test_pca_inverse_transform(n_components, data, expected_transformed, expected_inverse):
    pca = PCA(n_components)
    transformed = pca.fit_transform(data)
    inversed = pca.inverse_transform(transformed)
    assert np.array_equal(transformed, expected_transformed)
    assert np.array_equal(inversed, expected_inverse)


# Helper function to create test images
def create_reference_image(shape, dtype=np.uint8):
    if dtype == np.uint8:
        return np.random.randint(0, 256, shape, dtype=dtype)
    else:
        return np.random.rand(*shape).astype(dtype)

@pytest.mark.parametrize("img_shape, ref_shape, dtype", [
    ((100, 100, 3), (100, 100, 3), np.uint8),
    ((100, 100), (100, 100), np.uint8),
    ((50, 50, 3), (100, 100, 3), np.uint8),
    ((100, 100, 3), (50, 50, 3), np.uint8),
    ((100, 100, 3), (50, 50, 3), np.float32),
])
def test_apply_histogram_shapes_and_types(img_shape, ref_shape, dtype):
    img = create_reference_image(img_shape, dtype)
    reference_image = create_reference_image(ref_shape, dtype)
    blend_ratio = 0.5

    result = apply_histogram(img, reference_image, blend_ratio)

    assert result.shape == img_shape
    assert result.dtype == dtype

@pytest.mark.parametrize("blend_ratio", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_apply_histogram_blend_ratio(blend_ratio):
    img = create_reference_image((100, 100, 3))
    reference_image = create_reference_image((100, 100, 3))

    result = apply_histogram(img, reference_image, blend_ratio)

    if blend_ratio == 0.0:
        np.testing.assert_array_equal(result, img)
    elif blend_ratio == 1.0:
        assert not np.array_equal(result, img)
    else:
        assert not np.array_equal(result, img)
        assert not np.array_equal(result, reference_image)

def test_apply_histogram_grayscale():
    img = create_reference_image((100, 100))
    reference_image = create_reference_image((100, 100))
    blend_ratio = 0.5

    result = apply_histogram(img, reference_image, blend_ratio)

    assert result.shape == (100, 100)
    assert len(result.shape) == 2  # Ensure it remains grayscale

def test_apply_histogram_multichannel():
    img = create_reference_image((100, 100, 3))
    reference_image = create_reference_image((100, 100, 3))
    blend_ratio = 0.5

    result = apply_histogram(img, reference_image, blend_ratio)

    assert result.shape == (100, 100, 3)
    assert len(result.shape) == 3  # Ensure it remains multichannel

def test_apply_histogram_resize():
    img = create_reference_image((100, 100, 3))
    reference_image = create_reference_image((50, 50, 3))
    blend_ratio = 0.5

    result = apply_histogram(img, reference_image, blend_ratio)

    assert result.shape == (100, 100, 3)

@pytest.mark.parametrize("img_shape, ref_shape", [
    ((100, 100, 3), (100, 100, 3)),
    ((100, 100), (100, 100)),
    ((50, 50, 3), (100, 100, 3)),
])
def test_apply_histogram_preserves_range(img_shape, ref_shape):
    img = create_reference_image(img_shape)
    reference_image = create_reference_image(ref_shape)
    blend_ratio = 0.5

    result = apply_histogram(img, reference_image, blend_ratio)

    assert result.min() >= 0
    assert result.max() <= 255

def test_apply_histogram_float_input():
    img = create_reference_image((100, 100, 3), dtype=np.float32)
    reference_image = create_reference_image((100, 100, 3), dtype=np.float32)
    blend_ratio = 0.5

    result = apply_histogram(img, reference_image, blend_ratio)

    assert result.dtype == np.float32
    assert 0 <= result.min() <= result.max() <= 1

def test_apply_histogram_different_distributions():
    img = np.full((100, 100, 3), 50, dtype=np.uint8)
    reference_image = np.full((100, 100, 3), 200, dtype=np.uint8)
    blend_ratio = 1.0

    result = apply_histogram(img, reference_image, blend_ratio)

    assert result.mean() > img.mean()

def test_apply_histogram_identity():
    img = create_reference_image((100, 100, 3))
    blend_ratio = 1.0

    result = apply_histogram(img, img, blend_ratio)

    np.testing.assert_array_almost_equal(result, img)
