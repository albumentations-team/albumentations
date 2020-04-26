import numpy as np


def scale(points, scale_factor=(1, 1, 1)):
    transformation_matrix = np.eye(3)
    np.fill_diagonal(transformation_matrix, scale_factor)
    points[:, :3] = np.dot(points[:, :3], transformation_matrix)
    return points


def rotate_around_axis(points, axis, angle):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by angle in radians.
    https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    """
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rotation_matrix = np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )
    points[:, :3] = np.dot(points[:, :3], rotation_matrix.T)
    return points


def crop(points, x_min, y_min, z_min, x_max, y_max, z_max):
    if x_max <= x_min or y_max <= y_min or z_max <= z_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max and z_min < z_max. But we got"
            " (x_min = {x_min}, y_min = {y_min}, z_min = {z_min},"
            " x_max = {x_max}, y_max = {y_max}, z_max = {z_max})".format(
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                z_min=z_min,
                z_max=z_max,
            )
        )
    inds = np.all(
        [
            (points[:, 0] >= x_min),
            (points[:, 0] < x_max),
            (points[:, 1] >= y_min),
            (points[:, 1] < y_max),
            (points[:, 2] >= z_min),
            (points[:, 2] < z_max),
        ],
        axis=0,
    )
    return inds


def center(points, origin=(0, 0, 0)):
    points[:, :3] -= origin + points[:, :3].mean(axis=0)
    return points


def move(points, offset=(0, 0, 0)):
    points[:, :3] = points[:, :3] + offset
    return points
