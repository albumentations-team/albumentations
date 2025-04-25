import cv2
import numpy as np

import albumentations as A


transforms2metadata_key = {
    A.FDA: "fda_metadata",
    A.HistogramMatching: "hm_metadata",
    A.PixelDistributionAdaptation: "pda_metadata",
}

AUGMENTATION_CLS_PARAMS = [
    [
        A.ImageCompression,
        {
            "quality_range": (10, 80),
            "compression_type": "webp",
        },
    ],
    [
        A.HueSaturationValue,
        {"hue_shift_limit": 70, "sat_shift_limit": 95, "val_shift_limit": 55},
    ],
    [A.RGBShift, {"r_shift_limit": 70, "g_shift_limit": 80, "b_shift_limit": 40}],
    [A.RandomBrightnessContrast, {"brightness_limit": 0.5, "contrast_limit": 0.8}],
    [A.Blur, {"blur_limit": (3, 5)}],
    [A.MotionBlur, {"blur_limit": (3, 5)}],
    [A.MedianBlur, {"blur_limit": (3, 5)}],
    [A.GaussianBlur, {"blur_limit": (3, 5)}],
    [
        A.GaussNoise,
        {"std_range": (0.2, 0.44), "mean_range": (0.0, 0.0), "per_channel": False},
    ],
    [A.CLAHE, {"clip_limit": 2, "tile_grid_size": (12, 12)}],
    [A.RandomGamma, {"gamma_limit": (10, 90)}],
    [
        A.CoarseDropout,
        [{
            "num_holes_range": (2, 5),
            "hole_height_range": (3, 4),
            "hole_width_range": (4, 6),
        },
        {
            "num_holes_range": (2, 5),
            "hole_height_range": (0.1, 0.2),
            "hole_width_range": (0.2, 0.3),
        }]
    ],
    [
        A.RandomSnow,
        {"snow_point_range": (0.2, 0.4), "brightness_coeff": 4},
    ],
    [
        A.RandomRain,
        {
            "slant_range": (-5, 5),
            "drop_length": 15,
            "drop_width": 2,
            "drop_color": (100, 100, 100),
            "blur_value": 3,
            "brightness_coefficient": 0.5,
            "rain_type": "heavy",
        },
    ],
    [A.RandomFog, {"fog_coef_range": (0.2, 0.8), "alpha_coef": 0.11}],
    [
        A.RandomSunFlare,
        {
            "flare_roi": (0.1, 0.1, 0.9, 0.6),
            "angle_range": (0.1, 0.95),
            "num_flare_circles_range": (7, 11),
            "src_radius": 300,
            "src_color": (200, 200, 200),
        },
    ],
    [
        A.RandomGravel,
        {
            "gravel_roi": (0.1, 0.4, 0.9, 0.9),
            "number_of_patches": 2,
        },
    ],
    [
        A.RandomShadow,
        {
            "shadow_roi": (0.1, 0.4, 0.9, 0.9),
            "num_shadows_limit": (2, 4),
            "shadow_dimension": 8,
        },
    ],
    [
        A.PadIfNeeded,
        {
            "min_height": 512,
            "min_width": 512,
            "border_mode": cv2.BORDER_CONSTANT,
            "fill": (10, 10, 10),
        },
    ],
    [
        A.Rotate,
        {
            "limit": 120,
            "interpolation": cv2.INTER_CUBIC,
            "border_mode": cv2.BORDER_CONSTANT,
            "fill": (10, 10, 10),
            "crop_border": False,
        },
    ],
    [
        A.SafeRotate,
        {
            "limit": 120,
            "interpolation": cv2.INTER_CUBIC,
            "border_mode": cv2.BORDER_CONSTANT,
            "fill": 10,
        },
    ],
    [
        A.ShiftScaleRotate,
        [{
            "shift_limit": (-0.2, 0.2),
            "scale_limit": (-0.2, 0.2),
            "rotate_limit": (-70, 70),
            "interpolation": cv2.INTER_CUBIC,
            "border_mode": cv2.BORDER_CONSTANT,
            "fill": 10,
        },
        {
            "shift_limit_x": (-0.3, 0.3),
            "shift_limit_y": (-0.4, 0.4),
            "scale_limit": (-0.2, 0.2),
            "rotate_limit": (-70, 70),
            "interpolation": cv2.INTER_CUBIC,
            "border_mode": cv2.BORDER_CONSTANT,
            "fill": 10,
        }]
    ],
    [
        A.OpticalDistortion,
        {
            "distort_limit": 0.2,
            "interpolation": cv2.INTER_AREA,
        },
    ],
    [
        A.GridDistortion,
        {
            "num_steps": 10,
            "distort_limit": 0.5,
            "interpolation": cv2.INTER_CUBIC,
        },
    ],
    [
        A.ElasticTransform,
        {
            "alpha": 2,
            "sigma": 25,
            "interpolation": cv2.INTER_CUBIC,
        },
    ],
    [A.CenterCrop, {"height": 90, "width": 95}],
    [A.RandomCrop, {"height": 90, "width": 95}],
    [A.AtLeastOneBBoxRandomCrop, {"height": 90, "width": 95}],
    [A.CropNonEmptyMaskIfExists, {"height": 10, "width": 10}],
    [A.RandomSizedCrop, {"min_max_height": (90, 100), "size": (90, 90)}],
    [A.Crop, {"x_max": 64, "y_max": 64}],
    [A.ToFloat, {"max_value": 16536}],
    [
        A.Normalize,
        {
            "mean": (0.385, 0.356, 0.306),
            "std": (0.129, 0.124, 0.125),
            "max_pixel_value": 100.0,
        },
    ],
    [A.RandomScale, {"scale_limit": 0.2, "interpolation": cv2.INTER_CUBIC}],
    [A.Resize, {"height": 64, "width": 64}],
    [A.SmallestMaxSize, {"max_size": 64, "interpolation": cv2.INTER_NEAREST}],
    [A.LongestMaxSize, [{"max_size": 128, "interpolation": cv2.INTER_NEAREST},
                        {"max_size_hw": (127, 126)}]],
    [A.RandomGridShuffle, {"grid": (4, 4)}],
    [A.Solarize, {"threshold_range": [0.5, 0.5]}],
    [A.Posterize, {"num_bits": (3, 5)}],
    [A.Equalize, {"mode": "pil", "by_channels": False}],
    [
        A.MultiplicativeNoise,
        {"multiplier": (0.7, 2.3), "per_channel": True, "elementwise": True},
    ],
    [
        A.ColorJitter,
        {
            "brightness": [0.2, 0.3],
            "contrast": [0.7, 0.9],
            "saturation": [1.2, 1.7],
            "hue": [-0.2, 0.1],
        },
    ],
    [
        A.Perspective,
        {
            "scale": 0.5,
            "keep_size": True,
            "border_mode": cv2.BORDER_REFLECT_101,
            "fill": 10,
            "fill_mask": 100,
            "fit_output": True,
            "interpolation": cv2.INTER_CUBIC,
        },
    ],
    [A.Sharpen, {"alpha": [0.2, 0.5], "lightness": [0.5, 1.0]}],
    [A.Emboss, {"alpha": [0.2, 0.5], "strength": [0.5, 1.0]}],
    [A.RandomToneCurve, {"scale": 0.2, "per_channel": False}],
    [A.RandomToneCurve, {"scale": 0.3, "per_channel": True}],
    [
        A.CropAndPad,
        {
            "px": 10,
            "keep_size": False,
            "sample_independently": False,
            "interpolation": cv2.INTER_CUBIC,
            "fill_mask": [10, 20, 30],
            "fill": [11, 12, 13],
            "border_mode": cv2.BORDER_REFLECT101,
        },
    ],
    [
        A.Superpixels,
        {
            "p_replace": (0.5, 0.7),
            "n_segments": (20, 30),
            "max_size": 25,
            "interpolation": cv2.INTER_CUBIC,
        },
    ],
    [
        A.Affine,
        [{
            "scale": 0.5,
            "translate_percent": 0.1,
            "translate_px": None,
            "rotate": 33,
            "shear": 21,
            "interpolation": cv2.INTER_CUBIC,
            "fill": 25,
            "fill_mask": 0,
            "border_mode": cv2.BORDER_CONSTANT,
            "fit_output": False,
        },
        {
            "scale": {"x": [0.3, 0.5], "y": [0.1, 0.2]},
            "translate_percent": None,
            "translate_px": {"x": [10, 20], "y": [5, 10]},
            "rotate": [333, 360],
            "shear": {"x": [31, 38], "y": [41, 48]},
            "interpolation": 3,
            "fill": [10, 20, 30],
            "fill_mask": 1,
            "border_mode": cv2.BORDER_REFLECT,
            "fit_output": False,
        }
        ]
    ],
    [
        A.PiecewiseAffine,
        {
            "scale": 0.33,
            "nb_rows": (10, 20),
            "nb_cols": 33,
            "interpolation": cv2.INTER_AREA,
            "mask_interpolation": cv2.INTER_NEAREST,
            "absolute_scale": True,
        },
    ],
    [A.ChannelDropout, dict(channel_drop_range=(1, 2), fill=1)],
    [A.ChannelShuffle, {}],
    [A.Downscale, dict(scale_range=[0.5, 0.75], interpolation_pair={
        "downscale": cv2.INTER_LINEAR,
        "upscale": cv2.INTER_LINEAR,
    })],
    [A.FromFloat, dict(dtype="uint8", max_value=1)],
    [A.HorizontalFlip, {}],
    [A.ISONoise, dict(color_shift=(0.2, 0.3), intensity=(0.7, 0.9))],
    [A.InvertImg, {}],
    [A.MaskDropout, dict(max_objects=2, fill=0, fill_mask=0)],
    [A.NoOp, {}],
    [
        A.RandomResizedCrop,
        dict(size=(20, 30), scale=(0.5, 0.6), ratio=(0.8, 0.9)),
    ],
    [A.FancyPCA, dict(alpha=0.3)],
    [A.RandomRotate90, {}],
    [A.ToGray, {"method": "pca"}],
    [A.ToRGB, {}],
    [A.ToSepia, {}],
    [A.Transpose, {}],
    [A.VerticalFlip, {}],
    [A.RingingOvershoot, dict(blur_limit=(7, 15), cutoff=(np.pi / 5, np.pi / 2))],
    [
        A.UnsharpMask,
               {
            "blur_limit": (3, 7),  # Allow for stronger blur
            "sigma_limit": (0.5, 2.0),  # Increase sigma range
            "alpha": (0.5, 1.0),  # Allow for stronger sharpening
            "threshold": 5,  # Lower threshold to allow more changes
        },
    ],
    [A.AdvancedBlur, dict(blur_limit=(3, 5), rotate_limit=(60, 90))],
    [A.PixelDropout, [{"dropout_prob": 0.1, "per_channel": True, "drop_value": None},
                         {
                            "dropout_prob": 0.1,
                            "per_channel": False,
                            "drop_value": 2,
                            "mask_drop_value": 15,
        },
                      ],
     ],
    [
        A.RandomCropFromBorders,
        dict(crop_left=0.2, crop_right=0.3, crop_top=0.05, crop_bottom=0.5),
    ],
    [
        A.Spatter,
        [
            dict(
            mode="rain",
            mean=(0.65, 0.65),
            std=(0.3, 0.3),
            gauss_sigma=(2, 2),
            cutout_threshold=(0.68, 0.68),
            intensity=(0.6, 0.6),
        ),
        dict(
            mode="mud",
            mean=(0.65, 0.65),
            std=(0.3, 0.3),
            gauss_sigma=(2, 2),
            cutout_threshold=(0.68, 0.68),
            intensity=(0.6, 0.6),
        )
    ],
    ],
    [
        A.ChromaticAberration,
        dict(
            primary_distortion_limit=0.02,
            secondary_distortion_limit=0.05,
            mode="green_purple",
            interpolation=cv2.INTER_LINEAR,
        ),
    ],
    [A.Defocus, {"radius": (5, 7), "alias_blur": (0.2, 0.6)}],
    [A.ZoomBlur, {"max_factor": (1.56, 1.7), "step_factor": (0.02, 0.04)}],
    [
        A.XYMasking,
        {
            "num_masks_x": (1, 3),
            "num_masks_y": 3,
            "mask_x_length": (10, 20),
            "mask_y_length": 10,
            "fill_mask": 1,
            "fill": 0,
        },
    ],
    [
        A.PadIfNeeded,
        {
            "min_height": 512,
            "min_width": 512,
            "border_mode": 0,
            "fill": [124, 116, 104],
            "position": "top_left",
        },
    ],
    [A.GlassBlur, dict(sigma=0.8, max_delta=5, iterations=3, mode="exact")],
    [
        A.GridDropout,
        dict(
            ratio=0.75,
            holes_number_xy=(2, 10),
            shift_xy=(10, 20),
            random_offset=True,
            fill=10,
            fill_mask=20,
        ),
    ],
    [A.Morphological, {}],
    [A.D4, {}],
    [A.SquareSymmetry, {}],
    [A.PlanckianJitter, {}],
    [A.OverlayElements, {}],
    [A.RandomCropNearBBox, {}],
    [
        A.TextImage,
        dict(
            font_path="./tests/files/LiberationSerif-Bold.ttf",
            font_size_fraction_range=(0.8, 0.9),
            font_color=(255, 0, 0),  # red in RGB
            stopwords=(
                "a",
                "the",
                "is",
                "of",
                "it",
                "and",
                "to",
                "in",
                "on",
                "with",
                "for",
                "at",
                "by",
            ),
        ),
    ],
    [A.GridElasticDeform, {"num_grid_xy": (10, 10), "magnitude": 10}],
    [A.ShotNoise, {"scale_range": (0.1, 0.3)}],
    [A.TimeReverse, {}],
    [A.TimeMasking, {"time_mask_param": 10}],
    [A.FrequencyMasking, {"freq_mask_param": 30}],
    [A.Pad, {"padding": 10}],
    [A.Erasing, {}],
    [A.AdditiveNoise, {}],
    [A.SaltAndPepper, {"amount": (0.5, 0.5), "salt_vs_pepper": (0.5, 0.5)}],
    [A.PlasmaBrightnessContrast, {"brightness_range": (0.5, 0.5), "contrast_range": (0.5, 0.5)}],
    [A.PlasmaShadow, {}],
    [A.Illumination, {}],
    [A.ThinPlateSpline, {}],
    [A.AutoContrast, [
        {"cutoff": 0, "ignore": None, "method": "cdf"},
        {"cutoff": 0, "ignore": None, "method": "pil"},
    ]],
    [A.PadIfNeeded3D, {"min_zyx": (300, 200, 400), "pad_divisor_zyx": (10, 10, 10), "position": "center", "fill": 10, "fill_mask": 20}],
    [A.Pad3D, {"padding": 10}],
    [A.CenterCrop3D, {"size": (2, 30, 30)}],
    [A.RandomCrop3D, {"size": (2, 30, 30)}],
    [A.CoarseDropout3D, {"num_holes_range": (1, 3), "hole_depth_range": (0.1, 0.2), "hole_height_range": (0.1, 0.2), "hole_width_range": (0.1, 0.2), "fill": 0, "fill_mask": None}],
    [A.CubicSymmetry, {}],
    [A.AtLeastOneBBoxRandomCrop, {"height": 80, "width": 80, "erosion_factor": 0.2}],
    [A.ConstrainedCoarseDropout, {"num_holes_range": (1, 3), "hole_height_range": (0.1, 0.2), "hole_width_range": (0.1, 0.2), "fill": 0, "fill_mask": 0, "mask_indices": [1]}],
    [A.RandomSizedBBoxSafeCrop, {"height": 80, "width": 80, "erosion_rate": 0.2}],
    [A.HEStain, [
        {"method": "vahadane", "intensity_scale_range": (0.5, 1.5), "intensity_shift_range": (-0.1, 0.1), "augment_background": False},
        {"method": "macenko", "intensity_scale_range": (0.5, 1.5), "intensity_shift_range": (-0.1, 0.1), "augment_background": True},
        {"method": "random_preset",
         "intensity_scale_range": (0.5, 1.5), "intensity_shift_range": (-0.1, 0.1), "augment_background": True},
    ]],
]
