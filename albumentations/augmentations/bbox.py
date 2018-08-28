__all__ = ['normalize_bbox', 'denormalize_bbox', 'convert_bbox_to_albumentations', 'convert_bbox_from_albumentations',
           'convert_bboxes_to_albumentations', 'convert_bboxes_from_albumentations']


def normalize_bbox(bbox, cols, rows):
    x_min, y_min, x_max, y_max = bbox[:4]
    normalized_bbox = [x_min / cols, y_min / rows, x_max / cols, y_max / rows]
    return normalized_bbox + bbox[4:]


def denormalize_bbox(bbox, cols, rows):
    x_min, y_min, x_max, y_max = bbox[:4]
    denormalized_bbox = [x_min * cols, y_min * rows, x_max * cols, y_max * rows]
    return denormalized_bbox + bbox[4:]


def convert_bbox_to_albumentations(img, bbox, source_format):
    if source_format not in {'coco', 'pascal_voc'}:
        raise ValueError(
            "Unknown source_format {}. Supported formats are: 'coco' and 'pascal_voc'".format(source_format)
        )
    img_height, img_width = img.shape[:2]
    if source_format == 'coco':
        x_min, y_min, width, height = bbox[:4]
        x_max = x_min + width
        y_max = y_min + height
    else:
        x_min, y_min, x_max, y_max = bbox[:4]
    bbox = [x_min, y_min, x_max, y_max] + bbox[4:]
    bbox = normalize_bbox(bbox, img_width, img_height)
    return bbox


def convert_bbox_from_albumentations(img, bbox, target_format):
    if target_format not in {'coco', 'pascal_voc'}:
        raise ValueError(
            "Unknown target_format {}. Supported formats are: 'coco' and 'pascal_voc'".format(target_format)
        )
    img_height, img_width = img.shape[:2]
    bbox = denormalize_bbox(bbox, img_width, img_height)
    if target_format == 'coco':
        x_min, y_min, x_max, y_max = bbox[:4]
        width = x_max - x_min
        height = y_max - y_min
        bbox = [x_min, y_min, width, height] + bbox[4:]
    return bbox


def convert_bboxes_to_albumentations(img, bboxes, source_format):
    return [convert_bbox_to_albumentations(img, bbox, source_format) for bbox in bboxes]


def convert_bboxes_from_albumentations(img, bboxes, target_format):
    return [convert_bbox_from_albumentations(img, bbox, target_format) for bbox in bboxes]
