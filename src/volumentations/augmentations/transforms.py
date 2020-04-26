import random
import math

from ..core.transforms_interface import PointCloudsTransform, to_tuple
from . import functional as F

__all__ = [
    "Scale3d",
    "RotateAroundAxis3d",
    "Crop3d",
    "RandomMove3d",
    "Move3d",
    "Center3d",
    "RandomDropout3d",
    "Flip3d",
]


class Scale3d(PointCloudsTransform):
    """Scale the input point cloud.

    Args:
        scale_limit (float, float, float): maximum scaling of input point cloud.
            Default: (0.1, 0.1, 0.1).
        bias (list(float, float, float)): base scaling that is always applied.
            List of 3 values to determine the basic scaling. Default: (1, 1, 1).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        points

    """

    def __init__(
        self, scale_limit=(0.1, 0.1, 0.1), bias=(1, 1, 1), always_apply=False, p=0.5
    ):
        super().__init__(always_apply, p)
        self.scale_limit = []
        for limit, bias_for_axis in zip(scale_limit, bias):
            self.scale_limit.append(to_tuple(limit, bias=bias_for_axis))

    def get_params(self):
        scale = []
        for limit in self.scale_limit:
            scale.append(random.uniform(limit[0], limit[1]))
        return {"scale": scale}

    def apply(self, points, scale=(1, 1, 1), **params):
        return F.scale(points, scale)

    def apply_to_normals(self, normals, **params):
        return normals

    def apply_to_features(self, features, **params):
        return features

    def apply_to_camera(self, camera, **params):
        return camera

    def apply_to_labels(self, labels, **params):
        return labels

    def get_transform_init_args(self):
        return {"scale_limit": self.scale_limit}


class RotateAroundAxis3d(PointCloudsTransform):
    """Rotate point cloud around axis on random angle.

    Args:
        rotation_limit (float): maximum rotation of the input point cloud. Default: (pi / 2).
        axis (list(float, float, float)): axis around which the point cloud is rotated. Default: (0, 0, 1).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        points

    """

    def __init__(
        self, rotation_limit=math.pi / 2, axis=(0, 0, 1), always_apply=False, p=0.5
    ):
        super().__init__(always_apply, p)
        self.rotation_limit = to_tuple(rotation_limit, bias=0)
        self.axis = axis

    def get_params(self):
        angle = random.uniform(self.rotation_limit[0], self.rotation_limit[1])
        return {"angle": angle, "axis": self.axis}

    def apply(self, points, axis, angle, **params):
        return F.rotate_around_axis(points, axis, angle)

    def apply_to_normals(self, normals, axis, angle, **params):
        return F.rotate_around_axis(normals, axis, angle)

    def apply_to_bbox(self, bbox, axis, angle, **params):
        return F.rotate_around_axis(bbox, axis, angle)

    def apply_to_features(self, features, **params):
        return features

    def apply_to_camera(self, camera, **params):
        return camera

    def apply_to_labels(self, labels, **params):
        return labels

    def get_transform_init_args(self):
        return {
            "rotation_limit": to_tuple(self.rotation_limit, bias=0),
            "axis": self.axis,
        }


class Crop3d(PointCloudsTransform):
    """Crop region from image.

    Args:
        x_min (float): Minimum x coordinate.
        y_min (float): Minimum y coordinate.
        z_min (float): Minimum z coordinate.
        x_max (float): Maximum x coordinate.
        y_max (float): Maximum y coordinate.
        z_max (float): Maximum z coordinate.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        points

    """

    def __init__(
        self,
        x_min=-math.inf,
        y_min=-math.inf,
        z_min=-math.inf,
        x_max=math.inf,
        y_max=math.inf,
        z_max=math.inf,
        always_apply=False,
        p=1.0,
    ):
        super().__init__(always_apply, p)
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max

    @property
    def targets_as_params(self):
        return ["points"]

    def get_params_dependent_on_targets(self, params):
        return {
            "indexes": F.crop(
                params["points"],
                x_min=self.x_min,
                y_min=self.y_min,
                z_min=self.z_min,
                x_max=self.x_max,
                y_max=self.y_max,
                z_max=self.z_max,
            )
        }

    def apply(self, points, indexes, **params):
        return points[indexes]

    def apply_to_normals(self, normals, indexes, **params):
        return normals[indexes]

    def apply_to_labels(self, labels, indexes, **params):
        return labels[indexes]

    def apply_to_features(self, features, indexes, **params):
        return features[indexes]

    def apply_to_camera(self, camera, **params):
        return camera

    def get_transform_init_args_names(self):
        return ("x_min", "y_min", "z_min", "x_max", "y_max", "z_max")


class Center3d(PointCloudsTransform):
    """Move average of point cloud and move it to coordinate (0,0,0).

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        points

    """

    def __init__(self, offset=(0, 0, 0), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.offset = offset

    def apply(self, points, **params):
        return F.move(F.center(points), self.offset)

    def apply_to_normals(self, normals, **params):
        return normals

    def apply_to_features(self, features, **params):
        return features

    def apply_to_camera(self, camera, **params):
        return camera

    def apply_to_labels(self, labels, **params):
        return labels

    def get_transform_init_args(self):
        return {
            "offset": self.offset,
        }


class Move3d(PointCloudsTransform):
    """Move point cloud on offset.

    Args:
        offset (float): coorinate where to move origin of coordinate frame. Default: 0.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        points

    """

    def __init__(self, offset=(0, 0, 0), always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.offset = offset

    def get_params(self):
        return {"offset": self.offset}

    def apply(self, points, offset, **params):
        return F.move(points, offset)

    def apply_to_camera(self, camera, **params):
        return camera

    def apply_to_bbox(self, bbox, offset, **params):
        return F.move(bbox, offset)

    def apply_to_normals(self, normals, **params):
        return normals

    def apply_to_features(self, features, **params):
        return features

    def apply_to_labels(self, labels, **params):
        return labels

    def get_transform_init_args(self):
        return {
            "offset": self.offset,
        }


class RandomMove3d(Move3d):
    """Move point cloud on random offset.

    Args:
        x_min (float): Minimum x coordinate. Default: -1.
        y_min (float): Minimum y coordinate. Default: -1.
        z_min (float): Minimum z coordinate. Default: -1.
        x_max (float): Maximum x coordinate. Default: 1.
        y_max (float): Maximum y coordinate. Default: 1.
        z_max (float): Maximum z coordinate. Default: 1.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        points

    """

    def __init__(
        self,
        x_min=-1.0,
        y_min=-1.0,
        z_min=-1.0,
        x_max=1.0,
        y_max=1.0,
        z_max=1.0,
        offset=(0, 0, 0),
        always_apply=False,
        p=0.5,
    ):
        super().__init__(offset, always_apply, p)
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max

    def get_params(self):
        offset = [
            random.uniform(self.x_min, self.x_max),
            random.uniform(self.y_min, self.y_max),
            random.uniform(self.z_min, self.z_max),
        ]
        return {"offset": offset}

    def get_transform_init_args_names(self):
        return {
            "offset": self.offset,
            "x_min": self.x_min,
            "y_min": self.y_min,
            "z_min": self.z_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "z_max": self.z_max,
        }


class RandomDropout3d(PointCloudsTransform):
    """Randomly drop points from point cloud.

    Args:
        dropout_ratio (float): Percent of points to drop. Default: 0.2.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        points

    """

    def __init__(self, dropout_ratio=0.2, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.dropout_ratio = dropout_ratio

    @property
    def targets_as_params(self):
        return ["points"]

    def get_params_dependent_on_targets(self, params):
        points_len = len(params["points"])
        indexes = random.sample(
            range(points_len), k=int(points_len * (1 - self.dropout_ratio))
        )
        sorted_indexes = sorted(indexes)
        return {"indexes": sorted_indexes}

    def apply(self, points, indexes, **params):
        print(len(indexes), len(points))
        return points[indexes]

    def apply_to_normals(self, normals, indexes, **params):
        return normals[indexes]

    def apply_to_labels(self, labels, indexes, **params):
        return labels[indexes]

    def apply_to_features(self, features, indexes, **params):
        return features[indexes]

    def apply_to_camera(self, camera, **params):
        return camera

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def get_transform_init_args(self):
        return {"dropout_ratio": self.dropout_ratio}


class Flip3d(PointCloudsTransform):
    """Flip point cloud around axis
        Implemented as rotation on 180 deg around axis.

    Args:
        axis (list(float, float, float)): Axis to flip the point cloud around. Default: 0.2.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        points

    """

    def __init__(self, axis=(0, 0, 1), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.axis = axis

    def apply(self, points, **params):
        return F.rotate_around_axis(points, axis=self.axis, angle=math.pi)

    def apply_to_normals(self, normals, indexes, **params):
        return F.rotate_around_axis(normals, axis=self.axis, angle=math.pi)

    def apply_to_bbox(self, bbox, **params):
        return F.rotate_around_axis(bbox, axis=self.axis, angle=math.pi)

    def apply_to_camera(self, camera, **params):
        return camera

    def apply_to_features(self, features, **params):
        return features

    def apply_to_labels(self, labels, **params):
        return labels

    def get_transform_init_args(self):
        return {"axis": self.axis}
