import cv2
import imgaug as ia
import numpy as np
import pytest

from albumentations import Compose
from albumentations.augmentations.bbox_utils import (
    convert_bboxes_from_albumentations,
    convert_bboxes_to_albumentations,
)
import albumentations as A
from albumentations.imgaug.transforms import (
    IAAPiecewiseAffine,
    IAAFliplr,
    IAAFlipud,
    IAASuperpixels,
    IAASharpen,
    IAAAdditiveGaussianNoise,
    IAAPerspective,
    IAAAffine,
    IAACropAndPad,
)
from tests.utils import set_seed


TEST_SEEDS = (0, 1, 42, 111, 9999)


@pytest.mark.parametrize("augmentation_cls", [IAASuperpixels, IAASharpen, IAAAdditiveGaussianNoise])
def test_imgaug_image_only_augmentations(augmentation_cls, image, mask):
    aug = augmentation_cls(p=1)
    data = aug(image=image, mask=mask)
    assert data["image"].dtype == np.uint8
    assert data["mask"].dtype == np.uint8
    assert np.array_equal(data["mask"], mask)


@pytest.mark.parametrize("augmentation_cls", [IAAPiecewiseAffine, IAAPerspective])
def test_imgaug_dual_augmentations(augmentation_cls, image, mask):
    aug = augmentation_cls(p=1)
    data = aug(image=image, mask=mask)
    assert data["image"].dtype == np.uint8
    assert data["mask"].dtype == np.uint8


@pytest.mark.parametrize("augmentation_cls", [IAAPiecewiseAffine, IAAFliplr])
def test_imagaug_dual_augmentations_are_deterministic(augmentation_cls, image):
    aug = augmentation_cls(p=1)
    mask = np.copy(image)
    for _i in range(10):
        data = aug(image=image, mask=mask)
        assert np.array_equal(data["image"], data["mask"])


def test_imagaug_fliplr_transform_bboxes(image):
    aug = IAAFliplr(p=1)
    mask = np.copy(image)
    bboxes = [(10, 10, 20, 20), (20, 10, 30, 40)]
    expect = [(80, 10, 90, 20), (70, 10, 80, 40)]
    bboxes = convert_bboxes_to_albumentations(bboxes, "pascal_voc", rows=image.shape[0], cols=image.shape[1])
    data = aug(image=image, mask=mask, bboxes=bboxes)
    actual = convert_bboxes_from_albumentations(data["bboxes"], "pascal_voc", rows=image.shape[0], cols=image.shape[1])
    assert np.array_equal(data["image"], data["mask"])
    assert np.allclose(actual, expect)


def test_imagaug_flipud_transform_bboxes(image):
    aug = IAAFlipud(p=1)
    mask = np.copy(image)
    dummy_class = 1234
    bboxes = [(10, 10, 20, 20, dummy_class), (20, 10, 30, 40, dummy_class)]
    expect = [(10, 80, 20, 90, dummy_class), (20, 60, 30, 90, dummy_class)]
    bboxes = convert_bboxes_to_albumentations(bboxes, "pascal_voc", rows=image.shape[0], cols=image.shape[1])
    data = aug(image=image, mask=mask, bboxes=bboxes)
    actual = convert_bboxes_from_albumentations(data["bboxes"], "pascal_voc", rows=image.shape[0], cols=image.shape[1])
    assert np.array_equal(data["image"], data["mask"])
    assert np.allclose(actual, expect)


@pytest.mark.parametrize(
    ["aug", "keypoints", "expected"],
    [
        [IAAFliplr, [(20, 30, 0, 0)], [(80, 30, 0, 0)]],
        [IAAFliplr, [(20, 30, 45, 0)], [(80, 30, 45, 0)]],
        [IAAFliplr, [(20, 30, 90, 0)], [(80, 30, 90, 0)]],
        #
        [IAAFlipud, [(20, 30, 0, 0)], [(20, 70, 0, 0)]],
        [IAAFlipud, [(20, 30, 45, 0)], [(20, 70, 45, 0)]],
        [IAAFlipud, [(20, 30, 90, 0)], [(20, 70, 90, 0)]],
    ],
)
def test_keypoint_transform_format_xy(aug, keypoints, expected):
    transform = Compose([aug(p=1)], keypoint_params={"format": "xy", "label_fields": ["labels"]})

    image = np.ones((100, 100, 3))
    transformed = transform(image=image, keypoints=keypoints, labels=np.ones(len(keypoints)))
    assert np.allclose(expected, transformed["keypoints"])


@pytest.mark.parametrize(["aug", "keypoints", "expected"], [[IAAFliplr, [[20, 30, 0, 0]], [[79, 30, 0, 0]]]])
def test_iaa_transforms_emit_warning(aug, keypoints, expected):
    with pytest.warns(UserWarning, match="IAAFliplr transformation supports only 'xy' keypoints augmentation"):
        Compose([aug(p=1)], keypoint_params={"format": "xyas", "label_fields": ["labels"]})


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    [
        [A.IAASuperpixels, {}],
        [A.IAAAdditiveGaussianNoise, {}],
        [A.IAACropAndPad, {}],
        [A.IAAFliplr, {}],
        [A.IAAFlipud, {}],
        [A.IAAAffine, {}],
        [A.IAAPiecewiseAffine, {}],
        [A.IAAPerspective, {}],
    ],
)
@pytest.mark.parametrize("p", [0.5, 1])
@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("always_apply", (False, True))
def test_imgaug_augmentations_serialization(augmentation_cls, params, p, seed, image, mask, always_apply):
    aug = augmentation_cls(p=p, always_apply=always_apply, **params)
    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    set_seed(seed)
    ia.seed(seed)
    aug_data = aug(image=image, mask=mask)
    set_seed(seed)
    ia.seed(seed)
    deserialized_aug_data = deserialized_aug(image=image, mask=mask)
    assert np.array_equal(aug_data["image"], deserialized_aug_data["image"])
    assert np.array_equal(aug_data["mask"], deserialized_aug_data["mask"])


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    [
        [A.IAASuperpixels, {}],
        [A.IAAAdditiveGaussianNoise, {}],
        [A.IAACropAndPad, {}],
        [A.IAAFliplr, {}],
        [A.IAAFlipud, {}],
        [A.IAAAffine, {}],
        [A.IAAPiecewiseAffine, {}],
        [A.IAAPerspective, {}],
    ],
)
@pytest.mark.parametrize("p", [0.5, 1])
@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("always_apply", (False, True))
def test_imgaug_augmentations_for_bboxes_serialization(
    augmentation_cls, params, p, seed, image, albumentations_bboxes, always_apply
):
    aug = augmentation_cls(p=p, always_apply=always_apply, **params)
    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    set_seed(seed)
    ia.seed(seed)
    aug_data = aug(image=image, bboxes=albumentations_bboxes)
    set_seed(seed)
    ia.seed(seed)
    deserialized_aug_data = deserialized_aug(image=image, bboxes=albumentations_bboxes)
    assert np.array_equal(aug_data["image"], deserialized_aug_data["image"])
    assert np.array_equal(aug_data["bboxes"], deserialized_aug_data["bboxes"])


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    [
        [A.IAASuperpixels, {}],
        [A.IAAAdditiveGaussianNoise, {}],
        [A.IAACropAndPad, {}],
        [A.IAAFliplr, {}],
        [A.IAAFlipud, {}],
        [A.IAAAffine, {}],
        [A.IAAPiecewiseAffine, {}],
        [A.IAAPerspective, {}],
    ],
)
@pytest.mark.parametrize("p", [0.5, 1])
@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("always_apply", (False, True))
def test_imgaug_augmentations_for_keypoints_serialization(
    augmentation_cls, params, p, seed, image, keypoints, always_apply
):
    aug = augmentation_cls(p=p, always_apply=always_apply, **params)
    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    set_seed(seed)
    ia.seed(seed)
    aug_data = aug(image=image, keypoints=keypoints)
    set_seed(seed)
    ia.seed(seed)
    deserialized_aug_data = deserialized_aug(image=image, keypoints=keypoints)
    assert np.array_equal(aug_data["image"], deserialized_aug_data["image"])
    assert np.array_equal(aug_data["keypoints"], deserialized_aug_data["keypoints"])


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    [
        [IAAAffine, {"scale": 1.5}],
        [IAAPiecewiseAffine, {"scale": 1.5}],
        [IAAPerspective, {}],
    ],
)
def test_imgaug_transforms_binary_mask_interpolation(augmentation_cls, params):
    """Checks whether transformations based on DualTransform does not introduce a mask interpolation artifacts"""
    aug = augmentation_cls(p=1, **params)
    image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    mask = np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)
    data = aug(image=image, mask=mask)
    assert np.array_equal(np.unique(data["mask"]), np.array([0, 1]))


def __test_multiprocessing_support_proc(args):
    x, transform = args
    return transform(image=x)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    [
        [IAAAffine, {"scale": 1.5}],
        [IAAPiecewiseAffine, {"scale": 1.5}],
        [IAAPerspective, {}],
    ],
)
def test_imgaug_transforms_multiprocessing_support(augmentation_cls, params, multiprocessing_context):
    """Checks whether we can use augmentations in multiprocessing environments"""
    aug = augmentation_cls(p=1, **params)
    image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)

    pool = multiprocessing_context.Pool(8)
    pool.map(__test_multiprocessing_support_proc, map(lambda x: (x, aug), [image] * 100))
    pool.close()
    pool.join()


@pytest.mark.parametrize(
    ["img_dtype", "px", "percent", "pad_mode", "pad_cval", "keep_size"],
    [
        [np.uint8, 10, None, cv2.BORDER_CONSTANT, 0, True],
        [np.uint8, -10, None, cv2.BORDER_CONSTANT, 0, True],
        [np.uint8, 10, None, cv2.BORDER_CONSTANT, 0, False],
        [np.uint8, -10, None, cv2.BORDER_CONSTANT, 0, False],
        [np.uint8, None, 0.1, cv2.BORDER_CONSTANT, 0, True],
        [np.uint8, None, -0.1, cv2.BORDER_CONSTANT, 0, True],
        [np.uint8, None, 0.1, cv2.BORDER_CONSTANT, 0, False],
        [np.uint8, None, -0.1, cv2.BORDER_CONSTANT, 0, False],
        [np.float32, None, 0.1, cv2.BORDER_CONSTANT, 0, False],
        [np.float32, None, -0.1, cv2.BORDER_CONSTANT, 0, False],
        [np.uint8, None, 0.1, cv2.BORDER_WRAP, 0, False],
        [np.uint8, None, 0.1, cv2.BORDER_REPLICATE, 0, False],
        [np.uint8, None, 0.1, cv2.BORDER_REFLECT101, 0, False],
    ],
)
def test_compare_crop_and_pad(img_dtype, px, percent, pad_mode, pad_cval, keep_size):
    h, w, c = 100, 100, 3
    mode_mapping = {
        cv2.BORDER_CONSTANT: "constant",
        cv2.BORDER_REPLICATE: "edge",
        cv2.BORDER_REFLECT101: "reflect",
        cv2.BORDER_WRAP: "wrap",
    }
    pad_mode_iaa = mode_mapping[pad_mode]

    bbox_params = A.BboxParams(format="pascal_voc")
    keypoint_params = A.KeypointParams(format="xy", remove_invisible=False)

    keypoints = np.random.randint(0, min(h, w), [10, 2])

    bboxes = []
    for i in range(10):
        x1, y1 = np.random.randint(0, min(h, w) - 2, 2)
        x2 = np.random.randint(x1 + 1, w - 1)
        y2 = np.random.randint(y1 + 1, h - 1)
        bboxes.append([x1, y1, x2, y2, 0])

    transform_albu = A.Compose(
        [
            A.CropAndPad(
                px=px,
                percent=percent,
                pad_mode=pad_mode,
                pad_cval=pad_cval,
                keep_size=keep_size,
                p=1,
                interpolation=cv2.INTER_AREA
                if (px is not None and px < 0) or (percent is not None and percent < 0)
                else cv2.INTER_LINEAR,
            )
        ],
        bbox_params=bbox_params,
        keypoint_params=keypoint_params,
    )
    transform_iaa = A.Compose(
        [IAACropAndPad(px=px, percent=percent, pad_mode=pad_mode_iaa, pad_cval=pad_cval, keep_size=keep_size, p=1)],
        bbox_params=bbox_params,
        keypoint_params=keypoint_params,
    )

    if img_dtype == np.uint8:
        img = np.random.randint(0, 256, (h, w, c), dtype=np.uint8)
    else:
        img = np.random.random((h, w, c)).astype(img_dtype)

    res_albu = transform_albu(image=img, keypoints=keypoints, bboxes=bboxes)
    res_iaa = transform_iaa(image=img, keypoints=keypoints, bboxes=bboxes)

    for key, item in res_albu.items():
        if key == "bboxes":
            bboxes = np.array(res_iaa[key])
            h = bboxes[:, 3] - bboxes[:, 1]
            w = bboxes[:, 2] - bboxes[:, 0]
            res_iaa[key] = bboxes[(h > 0) & (w > 0)]
        assert np.allclose(item, res_iaa[key]), f"{key} are not equal"
