import os
import sys
import matplotlib.pyplot as plt
import pycocotools
import numpy as np

from os import PathLike
from pathlib import Path
from collections import defaultdict, deque

import torch
from torch.serialization import safe_globals
import torchvision

sys.path.append(str(Path(__file__).parent.parent))

from torch_references.utils import *
from torch_references.coco_eval import *

######################
#                    #
######################


def get_eval_vals(test_results: dict):
    """
    <ol start="0">
    <li>AP@IoU=0.50:0.95 (all)</li>
    <li>AP@IoU=0.50 (all)</li>
    <li>AP@IoU=0.75 (all)</li>
    <li>AP@IoU=0.50:0.95 (small)</li>
    <li>AP@IoU=0.50:0.95 (medium)</li>
    <li>AP@IoU=0.50:0.95 (large)</li>
    <li>AR@IoU=0.50:0.95 (all)</li>
    <li>AR@IoU=0.50 (all)</li>
    <li>AR@IoU=0.75 (all)</li>
    <li>AR@IoU=0.50:0.95 (small)</li>
    <li>AR@IoU=0.50:0.95 (medium)</li>
    <li>AR@IoU=0.50:0.95 (large)</li>
    </ol>
    """
    arr = np.ndarray((12, len(test_results)))
    for i in range(len(test_results) - 1):
        result = test_results[i].coco_eval['bbox'].stats
        arr[:, i] = result
    result = test_results[-1].coco_eval['bbox'].stats
    arr[:, -1] = result
    return arr


######################
#                    #
######################


def visualize():
    read_image, get_transforms, model, device, get_class_name, draw_bounding_boxes = None, None, None, None, None, None
    image = read_image(
        "C:\\Users\\maxos\\OneDrive - rit.edu\\2245\\reu\\data\\Raw\\DJI_202503101234_015_llela1c\\DJI_20250310124714_0004_D.JPG")
    eval_transform = get_transforms(train=False)
    image_norm = image / 255.0

    # For inference
    model.to(device)
    model.eval()
    x = image / 255.0
    x = x.unsqueeze(0)
    x = x.to(device)

    with torch.no_grad():
        predictions = model(x)
        print(predictions)
        pred = predictions[0]

    # image = image[:3, ...]
    pred_labels = [f"{get_class_name(label)}: {score:.3f}" for label, score in zip(
        pred["labels"], pred["scores"])]
    pred_boxes = pred["boxes"].long()
    output_image = draw_bounding_boxes(
        image, pred_boxes, pred_labels, colors="red")

    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))
    plt.show()


def make_roc(save_dir: str | PathLike):
    pass


def make_pr(save_dir: str | PathLike):
    pass


def make_f1(save_dir: str | PathLike):
    pass


######################
#                    #
######################


def load_results_data(save_dir: str | PathLike):
    with safe_globals([MetricLogger, SmoothedValue, defaultdict, deque, CocoEvaluator, COCOeval, pycocotools.coco.COCO, list, np._core.multiarray.scalar, np.dtype, np.dtypes.Int64DType]):
        trained_results = torch.load(os.path.join(
            save_dir, "trained_results.pt"), weights_only=False)
        test_results = torch.load(os.path.join(
            save_dir, "test_results.pt"), weights_only=False)
    return trained_results, test_results


def make_graphs(save_dir: str | PathLike, trained_results: dict, test_results: dict):
    num_epochs = len(trained_results)
    eval_arr = get_eval_vals(test_results)
    print(trained_results[0])
    print(eval_arr)
    make_f1(save_dir)


######################
#                    #
######################


def main():
    save_dir = "C:\\Users\\maxos\\OneDrive - rit.edu\\2245\\reu\\results\\20250617_090103_e1_b2_lr0.001"
    trained_results, test_results = load_results_data(save_dir)
    make_graphs(save_dir, trained_results, test_results)


if __name__ == "__main__":
    main()
