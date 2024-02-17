import glob
import os
import random

import cv2
import numpy as np

from ...core.transforms_interface import ImageOnlyTransform
from . import functional as F


def load_label_data(label_file):
    label_data = {}
    label_data["labels"] = {}
    label_data["n_labels"] = {}
    with open(label_file) as f:
        for line in f:
            label = line.split()[0]
            if label not in label_data["labels"]:
                label_data["labels"][label] = {}
                label_data["n_labels"][label] = 1
            else:
                label_data["n_labels"][label] += 1

            numTag = label_data["n_labels"][label]
            xmin, ymin, xmax, ymax = line.split()[4:8]
            length = abs(float(xmin) - float(xmax))
            breadth = abs(float(ymin) - float(ymax))
            area = length * breadth

            truncation = line.split()[1]
            occlusion = line.split()[2]
            alpha = line.split()[3]
            threeDdim = line.split()[8:11]
            location = line.split()[11:14]
            rotationY = line.split()[14]

            label_data["labels"][label][f"l{numTag}"] = {
                "coord": [(int(float(xmin)), int(float(ymin))), (int(float(xmax)), int(float(ymax)))],
                "area": area,
                "Truncation": truncation,
                "Occlusion": occlusion,
                "Alpha": alpha,
                "ThreeDdim": threeDdim,
                "Location": location,
                "RotationY": rotationY,
            }

    return label_data


def generate_img(img, label_names, label_count, minority_label, new_label_file_name, parent_dir):
    mask = np.full(img.shape[:2], 255, dtype="uint8")
    for label_name in label_names:
        for tag in label_names[label_name]:
            coord = label_names[label_name][tag]["coord"]
            cv2.rectangle(mask, coord[0], coord[1], 0, -1)

    inv_mask = 255 - mask
    inPaint = cv2.inpaint(img, inv_mask, 3, cv2.INPAINT_NS)

    text = []

    for label in label_names:
        if label != minority_label:
            tags = list(label_names[label].keys())
            n_tags = label_count[label]
            n_add = label_count[minority_label]

            if n_tags > n_add:
                tag_add = random.sample(tags, n_add)
            else:
                tag_add = tags

            for tag in tag_add:
                set1 = " ".join(
                    [
                        label,
                        label_names[label][tag]["Truncation"],
                        label_names[label][tag]["Occlusion"],
                        label_names[label][tag]["Alpha"],
                    ]
                )
                set3 = " ".join(label_names[label][tag]["ThreeDdim"])
                set4 = " ".join(label_names[label][tag]["Location"])
                set5 = label_names[label][tag]["RotationY"]

                (xmin, ymin), (xmax, ymax) = label_names[label][tag]["coord"]
                set2 = " ".join([str(xmin), str(ymin), str(xmax), str(ymax)])
                inPaint[ymin:ymax, xmin:xmax, :] = img[ymin:ymax, xmin:xmax, :]

                line = " ".join([set1, set2, set3, set4, set5])
                text.append(line)
            print(f"Adding these tags of {label}:{tag_add}")
        else:
            minTags = list(label_names[label].keys())
            for tag in minTags:
                set1 = " ".join(
                    [
                        label,
                        label_names[label][tag]["Truncation"],
                        label_names[label][tag]["Occlusion"],
                        label_names[label][tag]["Alpha"],
                    ]
                )
                set3 = " ".join(label_names[label][tag]["ThreeDdim"])
                set4 = " ".join(label_names[label][tag]["Location"])
                set5 = label_names[label][tag]["RotationY"]
                (xmin, ymin), (xmax, ymax) = label_names[label][tag]["coord"]
                set2 = " ".join([str(xmin), str(ymin), str(xmax), str(ymax)])
                inPaint[ymin:ymax, xmin:xmax, :] = img[ymin:ymax, xmin:xmax, :]

                line = " ".join([set1, set2, set3, set4, set5])
                text.append(line)
    file_contents = "\n".join(text)
    with open(os.path.join(parent_dir, f"{new_label_file_name}.txt"), "w") as f:
        f.write(file_contents)

    return inPaint


def apply_earser(img, label_file, minority_label, new_label_file_name):
    try:
        if os.path.getsize(label_file) > 0:
            label_data = load_label_data(label_file)
            label_names = label_data["labels"]
            if minority_label not in label_names:
                raise RuntimeWarning(f"Given label file doesn't have {minority_label} label!")
            else:
                label_count = label_data["n_labels"]
                parent_dir = os.path.dirname(label_file)
                img = generate_img(img, label_names, label_count, minority_label, new_label_file_name, parent_dir)
        else:
            raise RuntimeError(f"{label_file} is empty!")
    except OSError as e:
        print(f"Error reading label file: {e}")

    return img


class MetaData(ImageOnlyTransform):
    """Class to extract data from labels of kitti format"""

    def __init__(self, always_apply=False, p=0.5):
        """
        Erase instances of labels in image.

        Args:
            label_file (_type_): _description_
            minority_label (_type_): _description_
            new_label_file_name (_type_): _description_
            always_apply (bool, optional): _description_. Defaults to False.
            p (float, optional): _description_. Defaults to 0.5.
        """
        super().__init__(always_apply=always_apply, p=p)

    def __call__(self, **data):
        self.label_file = data["label_file"]
        self.minority_label = data["minority_label"]
        self.new_label_file_name = data["new_label_file_name"]

        data["image"] = self.apply(**data)

        return data

    def apply(self, image, label_file, minority_label, new_label_file_name, **params):
        return apply_earser(image, label_file, minority_label, new_label_file_name)
