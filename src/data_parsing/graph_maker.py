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

from torch_references.coco_eval import *
from torch_references.utils import *


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
        "C:/Users/maxos/OneDrive - rit.edu/2245/reu/data/Raw/DJI_202503101234_015_llela1c/DJI_20250310124714_0004_D.JPG")
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


def make_lr(save_dir: str | PathLike, trained_results: dict[int, MetricLogger]):
    learning_rate = np.array([[i, trained_results[i].meters["lr"].avg]
                             for i in range(len(trained_results))])

    plt.figure(figsize=(4, 4))
    plt.title("Learning Rate per Epoch")
    plt.xlabel("Epoch Number")
    plt.ylabel("Learning Rate")

    plt.semilogy(learning_rate[:, 0], learning_rate[:, 1])

    plt.grid()
    
    plt.savefig(os.path.join(save_dir, "lr.png"), dpi=300, bbox_inches="tight")


# def make_results(save_dir: str | PathLike, trained_results: dict[int, MetricLogger], test_results: dict[int, CocoEvaluator]):
#     fig, axes = plt.subplots(2, 5, figsize=(15, 5), constrained_layout=True)

#     make_loss(trained_results, axes)
#     make_pr(test_results, axes)

#     plt.grid()
#     plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)

#     plt.savefig(os.path.join(save_dir, "results.png"), dpi=300, bbox_inches="tight")

def make_loss(save_dir: str | PathLike, trained_results: dict[int, MetricLogger]):
    fig, axes = plt.subplots(2, 2, figsize=(6, 6), constrained_layout=True)

    num_epochs = len(trained_results)
    loss_cls = np.zeros((num_epochs, 2))
    loss_box = np.zeros((num_epochs, 2))
    loss_obj = np.zeros((num_epochs, 2))
    loss_rpn_box = np.zeros((num_epochs, 2))
    for i in range(num_epochs):
        loss_box[i, :] = [i, trained_results[i].meters["loss"].avg]
        loss_obj[i, :] = [i, trained_results[i].meters["loss_box_reg"].avg]
        loss_cls[i, :] = [i, trained_results[i].meters["loss_classifier"].avg]
        loss_rpn_box[i, :] = [
            i, trained_results[i].meters["loss_rpn_box_reg"].avg]

    loss_box_fig = axes[0, 0]
    loss_box_fig.plot(loss_box[:, 0], loss_box[:, 1])
    loss_box_fig.set_title("Box Training Loss")
    loss_box_fig.set_xlabel("Epoch Number")
    loss_box_fig.set_ylabel("Box Loss")
    loss_box_fig.grid(True)

    loss_obj_fig = axes[0, 1]
    loss_obj_fig.plot(loss_obj[:, 0], loss_obj[:, 1])
    loss_obj_fig.set_title("Objectness Training Loss")
    loss_obj_fig.set_xlabel("Epoch Number")
    loss_obj_fig.set_ylabel("Objectness Loss")
    loss_obj_fig.grid(True)

    loss_cls_fig = axes[1, 0]
    loss_cls_fig.plot(loss_cls[:, 0], loss_cls[:, 1])
    loss_cls_fig.set_title("Classification Training Loss")
    loss_cls_fig.set_xlabel("Epoch Number")
    loss_cls_fig.set_ylabel("Classification Loss")
    loss_cls_fig.grid(True)

    loss_rpn_box_fig = axes[1, 1]
    loss_rpn_box_fig.plot(loss_rpn_box[:, 0], loss_rpn_box[:, 1])
    loss_rpn_box_fig.set_title("RPN Box Training Loss")
    loss_rpn_box_fig.set_xlabel("Epoch Number")
    loss_rpn_box_fig.set_ylabel("RPN Box Loss")
    loss_rpn_box_fig.grid(True)

    plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)

    plt.savefig(os.path.join(save_dir, "loss.png"),
                dpi=300, bbox_inches="tight")


def make_pr(save_dir: str | PathLike, test_results: dict[int, CocoEvaluator]):
    fig, axes = plt.subplots(2, 2, figsize=(6, 6), constrained_layout=True)

    # TxRxKxAxM
    # T = 0-10 [.5:.05:.95] iou thresholds                  => 0
    # R = 101 [0:.01:1] recall thresholds                   => ?
    # K = 3? # categories                                   => i
    # A = 4 areas for object (all, small, medium, large)    => 0
    # M = 3 [1 10 100] max detections threshold             => 2
    precisions_05 = np.array([test_results[i].coco_eval['bbox'].eval['precision'][0, 51, :, 0, 2]
                          for i in range(len(test_results))])
    precisions_05 = precisions_05.clip(min=0, max=None)
    recalls_05 = np.array([test_results[i].coco_eval['bbox'].eval['recall'][0, :, 0, 2]
                       for i in range(len(test_results))])
    recalls_05 = recalls_05.clip(min=0, max=None)
    mAP_05 = np.array([[i, test_results[i].coco_eval['bbox'].stats[1]]
                      for i in range(len(test_results))])
    mAP_05_095 = np.array([[i, test_results[i].coco_eval['bbox'].stats[0]]
                          for i in range(len(test_results))])

    precision_fig = axes[0, 0]
    plt.subplot(0, 0)
    for i in range(precisions_05.shape[1]):    
        precision_fig.plot(range(precisions_05.shape[0]), precisions_05[:, i])
    precision_fig.set_title("Precision@0.5 (R@0.5)")
    precision_fig.set_xlabel("Epoch Number")
    precision_fig.set_ylabel("Precision@0.5")
    precision_fig.grid(True)

    recall_fig = axes[0, 1]
    plt.subplot(0, 1)
    for i in range(recalls_05.shape[1]):    
        recall_fig.plot(range(recalls_05.shape[0]), recalls_05[:, i])
    recall_fig.set_title("Recall@0.5")
    recall_fig.set_xlabel("Epoch Number")
    recall_fig.set_ylabel("Recall@0.5")
    recall_fig.grid(True)
    
    map_05_fig = axes[1, 0]
    map_05_fig.plot(mAP_05[:, 0], mAP_05[:, 1])
    map_05_fig.set_title("mAP@0.5")
    map_05_fig.set_xlabel("Epoch Number")
    map_05_fig.set_ylabel("mAP@0.5")
    map_05_fig.grid(True)
    
    mAP_05_095_fig = axes[1, 1]
    mAP_05_095_fig.plot(mAP_05_095[:, 0], mAP_05_095[:, 1])
    mAP_05_095_fig.set_title("mAP@0.5:0.95")
    mAP_05_095_fig.set_xlabel("Epoch Number")
    mAP_05_095_fig.set_ylabel("mAP@0.5:0.95")
    mAP_05_095_fig.grid(True)

    plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)

    plt.savefig(os.path.join(save_dir, "pr.png"),
                dpi=300, bbox_inches="tight")


def make_roc(save_dir: str | PathLike, test_results: dict):
    # TxRxKxAxM
    # T = 0-10 [.5:.05:.95] iou thresholds                  => 0
    # R = 101 [0:.01:1] recall thresholds                   => ?
    # K = 3? # categories                                   => i
    # A = 4 areas for object (all, small, medium, large)    => 0
    # M = 3 [1 10 100] max detections threshold             => 2
    precisions = np.array([test_results[i].coco_eval['bbox'].eval['precision'] for i in range(len(test_results))])
    # recalls = np.array([test_results[i].coco_eval['bbox'].eval['recall'] for i in range(len(test_results))])



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
    # eval_arr = get_eval_vals(test_results)
    # print(eval_arr)
    save_dir = os.path.join(save_dir, "figures")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    make_lr(save_dir, trained_results)
    make_loss(save_dir, trained_results)
    make_pr(save_dir, test_results)
    make_roc(save_dir, test_results)

    print("Done")


######################
#                    #
######################


def main():
    save_dir = "C:/Users/maxos/OneDrive - rit.edu/2245/reu/results/training_out_8b"
    trained_results, test_results = load_results_data(save_dir)
    make_graphs(save_dir, trained_results, test_results)


if __name__ == "__main__":
    main()
