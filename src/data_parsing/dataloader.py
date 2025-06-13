#
# Loads/creates a dataset from privet data
#

import os
import pandas as pd

from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.tv_tensors import Image, BoundingBoxes, BoundingBoxFormat


class PrivetDataset(Dataset):
    """
    A `Dataset` object for fetching field images and their bounding boxes.
    """

    def __init__(self, img_dir: str, labels_dir: str, is_multispectral: bool = False, transform: Callable = None):
        """
        :param: img_dir: The directory containing all images/image subdirectories.
        :param: labels_dir: The directory containing all labels/label subdirectories.
        :param: is_multispectral: Whether or not the data is multispectral (i.e., a multichannel tensor) or "default" (RGB)
        :param: transform (optional): Any transforms to apply to the data.
        """
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.is_multispectral = is_multispectral
        self.transform = transform
        self.img_locs: dict[int, str] = {}
        self.labels_locs: dict[int, str] = {}
        self.classes: dict[int, str] = {}

        idx = 0
        for dirpath, dirnames, filenames in os.walk(img_dir):
            # skip the img_dir folder itself
            if dirpath == img_dir:
                continue 
            
            classes = {}
            
            # get the name of the folder containing the images (ex: Llela1c)
            terminal_dir = os.path.split(dirpath)[1]
            
            # get the class file for that set and extract the classes
            with open(os.path.join(labels_dir, terminal_dir, "classes.txt"), mode="r", encoding="utf-8") as class_file:
                class_num = 0
                for line in class_file:
                    class_name = self._convert_line_to_uniform_class(line)
                    classes[class_num] = class_name
                    class_num += 1
            
            # get the image/tensor representing the image
            for filename in filenames:
                if ".TIF" in filename: 
                    continue
                self.img_locs[idx] = os.path.join(dirpath, filename)
                if self.is_multispectral:
                    extension = ".JPG.pt"
                else:
                    extension = ".JPG"
                labels_filename = filename.replace(extension, ".txt")
                self.labels_locs[idx] = os.path.join(
                    labels_dir, terminal_dir, labels_filename)
                self.classes[idx] = classes
                idx += 1

    def _convert_line_to_uniform_class(self, line: str):
        """
        Converts a classname to one of three predetermined options, to make sure that
        class labels are uniform across class files.
        
        TODO: This function should be removed once the dataset is finalized and made uniform.
        """
        s = line.lower()
        if "privet" in s:
            return "privet"
        elif "yew" in s:
            return "yew"
        elif "path" in s:
            return "path"
        return "UNKNOWN"

    def get_is_multispectral(self):
        return self.is_multispectral

    def __len__(self):
        return len(self.img_locs)

    def _calculate_coords(self, hi: int, wi: int, cx: float, cy: float, hb: float, wb: float) -> tuple[float, float, float, float]:
        """
        Get the coordinates of a bounding box from YOLO format.
        :param: hi: the height of the image
        :param: wi: the width of the image
        :param: cx: the x-coordinate of the center of the bounding box
        :param: cy: the y-coordinate of the center of the bounding box
        :param: hb: the height of the bounding box
        :param: wb: the width of the bounding box
        """
        xmin = wi * (cx - wb / 2)
        xmax = wi * (cx + wb / 2)
        ymin = hi * (cy - hb / 2)
        ymax = hi * (cy + hb / 2)
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        return (xmin, ymin, xmax, ymax)

    def __getitem__(self, idx) -> tuple[Image, dict]:
        img_loc = self.img_locs[idx]
        labels_loc = self.labels_locs[idx]

        if self.is_multispectral:
            image = torch.load(img_loc)
        else:
            image = decode_image(img_loc) / torch.iinfo(torch.uint8).max
        if self.transform:
            image = self.transform(image)

        # Get bounding boxes and labels
        H = image.shape[1]
        W = image.shape[2]
        with open(labels_loc, mode="r", encoding="utf-8") as lf:
            lines = lf.readlines()

        N = len(lines)
        labels = torch.zeros((N), dtype=torch.uint8)
        boxes_tensor = torch.zeros((N, 4), dtype=torch.float16)
        areas = torch.zeros((N), dtype=torch.uint32)
        for i in range(N):
            label, cx, cy, wb, hb = map(float, lines[i].split())
            labels[i] = int(label)
            xmin, ymin, xmax, ymax = self._calculate_coords(H, W, cx, cy, hb, wb)
            boxes_tensor[i] = torch.tensor([xmin, ymin, xmax, ymax])
            areas[i] = int((xmax - xmin) * (ymax - ymin))
        boxes = BoundingBoxes(
            data=boxes_tensor, format=BoundingBoxFormat.XYXY, canvas_size=(H, W))

        return (
            image,
            {
                "boxes": boxes,
                "labels": labels,
                "image_id": idx,
                "area": areas,
                "iscrowd": torch.zeros((N)),
            }
        )


# Testing

def main():
    data = PrivetDataset("data/images", "data/labels")
    print(data[0])


if __name__ == "__main__":
    main()
