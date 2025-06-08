import pytest
import warnings
import albumentations as A


def test_from_dict_without_p_uses_correct_defaults():
    """Test that from_dict uses correct default p values when p is missing."""

    # Test transform with p=1.0 default (Resize)
    config = {
        "transform": {
            "__class_fullname__": "Resize",
            "height": 224,
            "width": 224
        }
    }

    with pytest.warns(UserWarning, match="Transform Resize has no 'p' parameter.*defaulting to 1"):
        transform = A.from_dict(config)

    assert transform.p == 1.0, "Resize should have p=1.0 when not specified"

    # Test transform with p=0.5 default (HorizontalFlip)
    config = {
        "transform": {
            "__class_fullname__": "HorizontalFlip"
        }
    }

    with pytest.warns(UserWarning, match="Transform HorizontalFlip has no 'p' parameter.*defaulting to 0.5"):
        transform = A.from_dict(config)

    assert transform.p == 0.5, "HorizontalFlip should have p=0.5 when not specified"


def test_compose_without_p_no_warning():
    """Test that Compose gets default p=1.0 when p is missing."""

    config = {
        "transform": {
            "__class_fullname__": "Compose",
            "transforms": []
        }
    }

    # Should not produce warning because Compose inherits from BaseCompose
    # which is handled specially in from_dict
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        transform = A.from_dict(config)

    # Compose actually does have a p parameter with default 1.0
    assert transform.p == 1.0


def test_nested_transforms_without_p():
    """Test that nested transforms get correct p values when missing."""

    config = {
        "transform": {
            "__class_fullname__": "Compose",
            "transforms": [
                {
                    "__class_fullname__": "Resize",
                    "height": 224,
                    "width": 224
                },
                {
                    "__class_fullname__": "RandomBrightnessContrast"
                },
                {
                    "__class_fullname__": "Normalize"
                }
            ]
        }
    }

    with pytest.warns(UserWarning) as warnings_list:
        transform = A.from_dict(config)

    # Should have 3 warnings for the 3 transforms missing p
    assert len(warnings_list) == 3

    # Check each transform has correct p value
    assert transform.transforms[0].p == 1.0  # Resize
    assert transform.transforms[1].p == 0.5  # RandomBrightnessContrast
    assert transform.transforms[2].p == 1.0  # Normalize


@pytest.mark.parametrize("transform_name,expected_p", [
    ("Resize", 1.0),
    ("PadIfNeeded", 1.0),
    ("CenterCrop", 1.0),
    ("RandomCrop", 1.0),
    ("Normalize", 1.0),
    ("ToFloat", 1.0),
    ("FromFloat", 1.0),
    ("HorizontalFlip", 0.5),
    ("VerticalFlip", 0.5),
    ("RandomBrightnessContrast", 0.5),
    ("GaussNoise", 0.5),
    ("Blur", 0.5),
])
def test_various_transforms_default_p(transform_name, expected_p):
    """Test that various transforms get their correct default p values."""

    # Build minimal config for each transform
    config = {"transform": {"__class_fullname__": transform_name}}

    # Add required params for some transforms
    if transform_name == "Resize":
        config["transform"].update({"height": 224, "width": 224})
    elif transform_name == "PadIfNeeded":
        config["transform"].update({"min_height": 512, "min_width": 512})
    elif transform_name in ["CenterCrop", "RandomCrop"]:
        config["transform"].update({"height": 100, "width": 100})
    elif transform_name == "FromFloat":
        config["transform"].update({"dtype": "uint8"})

    with pytest.warns(UserWarning, match=f"Transform {transform_name} has no 'p' parameter"):
        transform = A.from_dict(config)

    assert transform.p == expected_p, f"{transform_name} should have p={expected_p} when not specified"


def test_transform_with_explicit_p_no_warning():
    """Test that no warning is produced when p is explicitly provided."""

    config = {
        "transform": {
            "__class_fullname__": "Resize",
            "height": 224,
            "width": 224,
            "p": 0.8  # Explicitly provided
        }
    }

    # Should not produce any warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        transform = A.from_dict(config)

    assert transform.p == 0.8
