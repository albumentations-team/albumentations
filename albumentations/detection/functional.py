def vflip_bbox(bbox, cols, rows):
    return (cols - bbox[0] - bbox[2], *bbox[1:])


def hflip_bbox(bbox, cols, rows):
    return (bbox[0], rows - bbox[1] - bbox[3], *bbox[2:])