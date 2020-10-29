import cv2

__all__ = ["read_bgr_image", "read_rgb_image"]


def read_bgr_image(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def read_rgb_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
