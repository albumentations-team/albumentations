import glob
import os
import random

import cv2
import numpy as np


class MetaData:
    """Class to extract data from labels of kitti format"""

    def __init__(self, parentDir, imgType):
        """Load parent directory and Image type present in the image directory

        Args:
            parentDir (string): Path to the directory that contains the Image folder and Label folder
            imgType (string): Image type (Ex: JPEG: jpg, PNG: png)
        """
        self.labelPath = os.path.join(parentDir, "labels")
        self.imgPath = os.path.join(parentDir, "images")
        self.imgType = imgType
        self.data = {}
        self.distribution = {}
        self.labelStrengthOrder = []

    def loadData(self):
        """Load data from the txt files and stores it as a dictionary.
        Also loads the distribution of classification classes present in the dataset.
        """
        for filePath in glob.glob(f"{self.labelPath}/*"):
            filename = os.path.basename(filePath).split(".")[0]
            if os.path.isfile(os.path.join(self.imgPath, f"{filename}.{self.imgType}")):
                self.data[filename] = {}
                self.data[filename]["labels"] = {}
                self.data[filename]["n_labels"] = {}
                with open(filePath) as f:
                    for line in f:
                        label = line.split()[0]
                        if label not in self.data[filename]["labels"].keys():
                            self.data[filename]["labels"][label] = {}
                            self.data[filename]["n_labels"][label] = 1
                        else:
                            self.data[filename]["n_labels"][label] += 1

                        numTag = self.data[filename]["n_labels"][label]
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

                        self.data[filename]["labels"][label][f"l{numTag}"] = {
                            "coord": [(int(float(xmin)), int(float(ymin))), (int(float(xmax)), int(float(ymax)))],
                            "area": area,
                            "Truncation": truncation,
                            "Occlusion": occlusion,
                            "Alpha": alpha,
                            "ThreeDdim": threeDdim,
                            "Location": location,
                            "RotationY": rotationY,
                        }

            for label in self.data[filename]["n_labels"]:
                if label not in self.distribution.keys():
                    self.distribution[label] = self.data[filename]["n_labels"][label]
                else:
                    self.distribution[label] += self.data[filename]["n_labels"][label]

    def identifyMinority(self):
        """Sorts the distribution of classification classes to return class with least number of detections and its strength.

        Returns:
            list: the first tells us the minority class and the second its number of occurences.
        """
        self.labelStrengthOrder = list(dict(sorted(self.distribution.items(), key=lambda item: item[1])).keys())
        minorityLabel = self.labelStrengthOrder[0]
        minorityStrength = self.distribution[minorityLabel]

        return [minorityLabel, minorityStrength]


def startEraser(parentDir, imgType, imgExpType):
    """Start the synthesis process of normalizing the classes and
    generates a synthesis image directory and a synthesis label directory.

    Args:
        parentDir (string): Path to parent directory that contains image and label directories.
        imgType (string): Type of image present in the Image directory
        imgExpType (string): Type of synthesised image to be generated
    """
    # currentDir = os.getcwd()
    # parentDir = os.path.dirname(currentDir)
    imgDir = os.path.join(parentDir, "images")
    synthImgDir = os.path.join(parentDir, "syntheticImages")
    synthLabDir = os.path.join(parentDir, "syntheticLabels")

    try:
        os.mkdir(synthImgDir)
        os.mkdir(synthLabDir)
    except FileExistsError:
        print("Synthetic dir already created!")

    obj1 = MetaData(parentDir, imgType)
    obj1.loadData()
    print(obj1.data)
    # print(obj1.distribution)
    minorityLabel, minorityStrength = obj1.identifyMinority()

    for filePath in glob.glob(f"{imgDir}/*"):
        filename = os.path.basename(filePath).split(".")[0]
        img = cv2.imread(filePath)
        mask = np.full(img.shape[:2], 255, dtype="uint8")

        labelData = obj1.data[filename]["labels"]
        labelCount = obj1.data[filename]["n_labels"]

        if minorityLabel not in labelData.keys():
            continue

        for labelname in labelData.keys():
            for tag in labelData[labelname]:
                coord = labelData[labelname][tag]["coord"]
                cv2.rectangle(mask, coord[0], coord[1], 0, -1)

        # cv.imshow("Mask", mask)
        # cv.waitKey(1000)
        invMask = 255 - mask
        # palette = cv2.bitwise_and(img, img, mask=mask)
        # print(palette)
        # print(invMask)
        # cv.imshow("Inverted Mask", invMask)
        # cv.waitKey(1000)
        inPaint = cv2.inpaint(img, invMask, 3, cv2.INPAINT_NS)
        # cv.imshow("Inpainted Pic", inPaint)
        # cv.waitKey(5000)
        text = []
        for label in labelData.keys():
            if label != minorityLabel:
                tags = list(labelData[label].keys())
                nTags = labelCount[label]
                nAdd = labelCount[minorityLabel]
                if nTags > nAdd:
                    tagAdd = random.sample(tags, nAdd)
                else:
                    tagAdd = tags
                for tag in tagAdd:
                    set1 = " ".join(
                        [
                            label,
                            labelData[label][tag]["Truncation"],
                            labelData[label][tag]["Occlusion"],
                            labelData[label][tag]["Alpha"],
                        ]
                    )
                    set3 = " ".join(labelData[label][tag]["ThreeDdim"])
                    set4 = " ".join(labelData[label][tag]["Location"])
                    set5 = labelData[label][tag]["RotationY"]

                    (xmin, ymin), (xmax, ymax) = labelData[label][tag]["coord"]
                    set2 = " ".join([str(xmin), str(ymin), str(xmax), str(ymax)])
                    inPaint[ymin:ymax, xmin:xmax, :] = img[ymin:ymax, xmin:xmax, :]

                    line = " ".join([set1, set2, set3, set4, set5])
                    text.append(line)
                    # cv.imshow("ROI", img[ymin:ymax, xmin:xmax, :])
                    # cv.waitKey(5000)
                print(f"Adding these tags of {label}:{tagAdd}")
            else:
                minTags = list(labelData[label].keys())
                for tag in minTags:
                    set1 = " ".join(
                        [
                            label,
                            labelData[label][tag]["Truncation"],
                            labelData[label][tag]["Occlusion"],
                            labelData[label][tag]["Alpha"],
                        ]
                    )
                    set3 = " ".join(labelData[label][tag]["ThreeDdim"])
                    set4 = " ".join(labelData[label][tag]["Location"])
                    set5 = labelData[label][tag]["RotationY"]
                    (xmin, ymin), (xmax, ymax) = labelData[label][tag]["coord"]
                    set2 = " ".join([str(xmin), str(ymin), str(xmax), str(ymax)])
                    inPaint[ymin:ymax, xmin:xmax, :] = img[ymin:ymax, xmin:xmax, :]

                    line = " ".join([set1, set2, set3, set4, set5])
                    text.append(line)
                    # cv.imshow("ROI", img[ymin:ymax, xmin:xmax, :])
                    # cv.waitKey(5000)

        # cv.imshow("Synthetic Image", inPaint)
        # cv.waitKey(5000)
        cv2.imwrite(os.path.join(synthImgDir, f"{filename}.{imgExpType}"), inPaint)
        print(f"Synthesised synthetic Image for {filename}.")
        fileContents = "\n".join(text)
        with open(os.path.join(synthLabDir, f"{filename}.txt"), "w") as f:
            f.write(fileContents)
        print(f"New label for synthesised {filename} written.")
