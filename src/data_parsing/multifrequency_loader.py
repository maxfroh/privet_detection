#
# Takes in raw images and creates a tensor containing all specified spectra and indices.
#
# Example usage:
#

import argparse
import os

from PIL import Image

import numpy as np

import torch
from torch import Tensor, tensor
from torchvision import transforms

NUM_CHANNELS = 14

######################
# Index Calculations #
######################


def SR(arr: Tensor, NIR_idx: int, R_idx: int) -> float:
    """
    Calculates the Simple Ratio (SR) between Near Infrared (NIR) and Red (R).
    """
    NIR = arr[NIR_idx, :, :]
    R = arr[R_idx, :, :]
    return NIR / R


def NDVI(arr: Tensor, NIR_idx: int, R_idx: int) -> float:
    """
    Calculates the Normalized Difference Vegetation Index (NDVI) using Near Infrared (NIR) and Red (R).
    """
    NIR = arr[NIR_idx, :, :]
    R = arr[R_idx, :, :]
    return (NIR - R) / (NIR + R)


def NDVI_RE(arr: Tensor, NIR_idx: int, RE_idx: int) -> float:
    """
    Calculates the Red Edge Normalized Difference Vegetation Index (NDVI) using Near Infrared (NIR) and Red Edge (RE).
    """
    NIR = arr[NIR_idx, :, :]
    RE = arr[RE_idx, :, :]
    return (NIR - RE) / (NIR + RE)


def GNDVI(arr: Tensor, NIR_idx: int, G_idx: int) -> float:
    """
    Calculates the Green Normalized Difference Vegetation Index (NDVI) using Near Infrared (NIR) and Green (G).
    """
    NIR = arr[NIR_idx, :, :]
    G = arr[G_idx, :, :]
    return (NIR - G) / (NIR + G)


def SAVI(arr: Tensor, NIR_idx: int, R_idx: int, L: float = 0.5) -> float:
    """
    Calculates the Soil Adjusted Vegetation Index (SAVI) using Near Infrared (NIR) and Red (R).
    L is a parameter that can be changed but is 0.5 by default.
    """
    NIR = arr[NIR_idx, :, :]
    R = arr[R_idx, :, :]
    return (NIR - R) / (NIR + R + L) * (1 + L)


def MSAVI2(arr: Tensor, NIR_idx: int, R_idx: int) -> float:
    """
    Calculates the Soil Adjusted Vegetation Index (SAVI) using Near Infrared (NIR) and Red (R).
    L is a parameter that can be changed but is 0.5 by default.
    """
    NIR = arr[NIR_idx, :, :]
    R = arr[R_idx, :, :]
    return (2 * NIR * torch.sqrt(torch.pow((2 * NIR + 1), 2) - 8 * (NIR - R))) / 2


def GEMI(arr: Tensor, NIR_idx: int, R_idx: int) -> float:
    """
    Calculates the Global Environmental Monitoring Index (GEMI) using Near Infrared (NIR) and Red (R).
    """
    NIR = arr[NIR_idx, :, :]
    R = arr[R_idx, :, :]
    eta = (2 * (torch.pow(NIR, 2) + torch.pow(R, 2)) +
           1.5 * NIR + 0.5 * R) / (NIR + R + 0.5)
    return eta * (1 - eta / 4) - (R - 0.125) / (1 - R)


def Cl_g(arr: Tensor, NIR_idx: int, G_idx: int) -> float:
    """
    Calculates the Green Chlorophyll Index (Cl_g) using Near Infrared (NIR) and Green (G).
    """
    NIR = arr[NIR_idx, :, :]
    G = arr[G_idx, :, :]
    return NIR / G - 1


def Cl_re(arr: Tensor, NIR_idx: int, RE_idx: int) -> float:
    """
    Calculates the Red Edge Chlorophyll Index (Cl_re) using Near Infrared (NIR) and Red Edge (RE).
    """
    NIR = arr[NIR_idx, :, :]
    RE = arr[RE_idx, :, :]
    return NIR / RE - 1

######################
#   File Functions   #
######################

def calculate_indices(arr: Tensor, channel: int=4):
    """
    Calculate all indices for the given tensor.
    
    Channels:
    | R | G | B | RE | NIR | SR | NDVI | NDVI_RE | GNDVI | SAVI | MSAVI2 | GEMI | Cl_g | Cl_re |
    """
    arr[channel, :, :] = SR(arr, 4, 0)
    channel += 1
    arr[channel, :, :] = NDVI(arr, 4, 0)
    channel += 1
    arr[channel, :, :] = NDVI_RE(arr, 4, 3)
    channel += 1
    arr[channel, :, :] = GNDVI(arr, 4, 1)
    channel += 1
    arr[channel, :, :] = SAVI(arr, 4, 0)
    channel += 1
    arr[channel, :, :] = MSAVI2(arr, 4, 0)
    channel += 1
    arr[channel, :, :] = GEMI(arr, 4, 0)
    channel += 1
    arr[channel, :, :] = Cl_g(arr, 4, 1)
    channel += 1
    arr[channel, :, :] = Cl_re(arr, 4, 3)
    channel += 1
    return arr

def process_dir(directory: str):
    print(f"Processing {directory}")
    full_root, folder_name = os.path.split(directory)
    root, image_folder = os.path.split(full_root)
    print(folder_name)
    ms_dir_name = os.path.join(root, "multispectral_tensors", folder_name)
    if not os.path.isdir(ms_dir_name):
        os.mkdir(ms_dir_name)
    files = [f for f in os.listdir(directory) if "_D." in f]
    for file in files:
        print(file)
        arr = process_file(os.path.join(directory, file))
        print(arr.shape)
        print(arr.dtype)
        filename = os.path.join(ms_dir_name, file) + ".pt"
        torch.save(arr.contiguous(), filename)

def process_file(file: str | os.PathLike[str]) -> Tensor:
    """
    Open a file and its other multispectral versions and return a tensor.
    """
    
    # Open D image and get RGB channels
    transform = transforms.ToTensor()
    try:
        img = Image.open(file)
        rgb_tensor = transform(img)
        w = img.width
        h = img.height
        arr = tensor(np.zeros((NUM_CHANNELS, h, w)), dtype=torch.float16)
        arr[0:3, :, :] = rgb_tensor
    except Exception as e:
        print(f"Failed because {e}")

    # open other channel images
    channel = 3
    formats = ["_MS_RE.TIF", "_MS_NIR.TIF"]
    files = [file.replace("_D.JPG", format) for format in formats]
    transform = transforms.Compose(
        [
            transforms.Resize(
                size=[h, w], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ]
    )
    for file in files:
        try:
            img = Image.open(file)
            c_tensor = transform(img)
            c_tensor = c_tensor / torch.iinfo(torch.uint16).max
            print(c_tensor.dtype)
            arr[channel, :, :] = c_tensor
            channel += 1
        except Exception as e:
            print(f"Failed because {e}")

    # calculate all indices for the image
    arr = calculate_indices(arr, channel)
    return arr

######################
# Program Operations #
######################


def parse_args():
    parser = argparse.ArgumentParser(
        prog="multifrequency_loader.py",
        description="Converts separate images with multiple frequencies into single tensor files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-dirs-in", nargs="+",
                        help="Image directory or directories")
    # parser.add_argument("-c", "--channels", default="all", help="all, D, or a custom list of frequencies and indices. Options: [R, G, B, RE, NIR, SR, NDVI, NDVI_RE, GNDVI, SAVI, MSAVI2, GEMI, Cl_g, Cl_re]")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    in_dirs = args.dirs_in
    for directory in in_dirs:
        process_dir(directory)
    print("Done.")


if __name__ == "__main__":
    main()
