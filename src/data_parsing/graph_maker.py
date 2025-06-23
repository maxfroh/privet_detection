import os
import sys
import re
import random
import argparse
import pycocotools
import matplotlib.pyplot as plt
import numpy as np

from os import PathLike
from pathlib import Path
from collections import defaultdict, deque
from operator import itemgetter

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.serialization import safe_globals
from torchvision.transforms import v2 as T
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes

sys.path.append(str(Path(__file__).parent.parent))

from torch_references.coco_eval import *
from torch_references.utils import *
from data_parsing.dataloader import PrivetDataset
from models.fast_rcnn import FasterRCNNResNet101


RAND_SEED = 7

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


def get_class_name(i):
    return {0: "bg", 1: "privet", 2: "yew"}[i]

def get_class_color(i):
    return {0: "black", 1: "red", 2: "salmon"}[i]


def get_transforms(train: bool = True):
    """
    Return transforms for the data.
    """
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(p=0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms=transforms)


def get_model(model: str, num_channels: int) -> torch.nn.Module:
    match model:
        case "faster_rcnn":
            return FasterRCNNResNet101(num_channels=num_channels)

def get_model_dir(save_dir: str | PathLike, model_name: str):
    return os.path.join(save_dir, "models",
                f"{model_name}.pt")

def get_best_model(save_dir, best_models):
    model = get_model("faster_rcnn", 3)
    best_model_name = sorted(best_models, key=itemgetter(1), reverse=True)[0][0]

    state_dict = torch.load(get_model_dir(save_dir, best_model_name), weights_only=False)["model_state_dict"]
    model.load_state_dict(state_dict)
    return model


######################
#                    #
######################


def make_img(plot, model):
    pass


def visualize(save_dir: str | PathLike, best_models: list[tuple[str, float]], test_data: DataLoader):
    model = get_best_model(save_dir, best_models)

    image = read_image(
        "C:/Users/mf0771/Documents/cut/images/privet1a_11/DJI_20250310120048_0547_D.JPG")
    eval_transform = get_transforms(train=False)
    image_norm = image / 255.0

    # For inference
    device = torch.accelerator.current_accelerator().type \
        if torch.accelerator.is_available() else "cpu"
    model.to(device)
    model.eval()
    x = image_norm
    x = x.unsqueeze(0)
    x = x.to(device)

    with torch.no_grad():
        x = eval_transform(x)
        predictions = model(x)
        pred = predictions[0]

    # image = image[:3, ...]
    pred_labels = [f"{get_class_name(label.item())}: {score.item():.4f}" for label, score in zip(
        pred["labels"], pred["scores"])]
    pred_boxes = pred["boxes"].long()
    colors = [get_class_color(label.item()) for label in pred["labels"]]
    output_image = draw_bounding_boxes(
        image, pred_boxes, pred_labels, colors=colors, width=10, font="arial.ttf", font_size=60, label_colors="white")

    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))
    # plt.show()
    plt.imsave(os.path.join(save_dir, "image2.png"), output_image.permute(1, 2, 0), dpi=300)


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
                          for i in range(len(test_results) - 1)])
    precisions_05 = precisions_05.clip(min=0, max=None)
    recalls_05 = np.array([test_results[i].coco_eval['bbox'].eval['recall'][0, :, 0, 2]
                       for i in range(len(test_results) - 1)])
    recalls_05 = recalls_05.clip(min=0, max=None)
    mAP_05 = np.array([[i, test_results[i].coco_eval['bbox'].stats[1]]
                      for i in range(len(test_results) - 1)])
    mAP_05_095 = np.array([[i, test_results[i].coco_eval['bbox'].stats[0]]
                          for i in range(len(test_results) - 1)])

    precision_fig = axes[0, 0]
    # plt.subplot(0, 0)
    for i in range(precisions_05.shape[1]):    
        precision_fig.plot(range(precisions_05.shape[0]), precisions_05[:, i])
    precision_fig.set_title("Precision@0.5 (R@0.5)")
    precision_fig.set_xlabel("Epoch Number")
    precision_fig.set_ylabel("Precision@0.5")
    precision_fig.grid(True)

    recall_fig = axes[0, 1]
    # plt.subplot(0, 1)
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

    make_lr(save_dir, trained_results)
    make_loss(save_dir, trained_results)
    make_pr(save_dir, test_results)
    # make_roc(save_dir, test_results)
    print("Done")


def make_graphs_and_vis(save_dir: str | PathLike, trained_results: dict, test_results: dict, best_models: dict[tuple[str, float]]=None, test_data: DataLoader=None):
    save_dir = os.path.join(save_dir, "figures")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    make_graphs(save_dir, trained_results, test_results)
    if best_models is not None and test_data is not None:
        visualize(save_dir, best_models, test_data)

######################
#                    #
######################

def setup_visualize(args, save_dir, test_results):
    batch_size = int(re.findall(r"(?<=b)\d{1,2}(?=_lr)", save_dir)[0])
    best_models = [f.replace(".pt", "") for f in os.listdir(os.path.join(save_dir, "models"))]
    indices = [(i, int(re.findall(r"\d{1,2}(?=_of_)", best_models[i])[0]) - 1) for i in range(len(best_models))]
    top_5_mAP = []
    for best_models_idx, actual_idx in indices:
        mAP = test_results[actual_idx].coco_eval['bbox'].stats[1]
        top_5_mAP.append((best_models[best_models_idx], mAP))

    is_multispectral = True if args.channels == "all" else False
    test_data = PrivetDataset(img_dir=args.img_dir, labels_dir=args.labels_dir,
                                 is_multispectral=is_multispectral)

    idxs = torch.randperm(len(test_data)).tolist()
    one_tenth = len(test_data) // 10
    eight_tenths = len(test_data) - (2 * one_tenth)
    test_data = torch.utils.data.Subset(test_data, idxs[(eight_tenths + one_tenth):])

    test_data = DataLoader(
        dataset=test_data, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return top_5_mAP, test_data

def parse_args():
    parser = argparse.ArgumentParser(
        prog="multifrequency_loader.py",
        description="Converts separate images with multiple frequencies into single tensor files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m", "--model", help="Which model to run.\nOptions: {faster_rcnn, }", default="faster_rcnn")
    parser.add_argument("-c", "--channels",
                        help="Which channels to use.\nOptions: {rgb, all}", default="rgb")
    parser.add_argument("--img_dir", help="The outer image directory")
    parser.add_argument("--labels_dir", help="The outer label directory")
    parser.add_argument("--save_dir", help="The directory results were saved in")

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)
    torch.random.manual_seed(RAND_SEED)
    torch.cuda.manual_seed_all(RAND_SEED)

    save_dir = args.save_dir

    trained_results, test_results = load_results_data(save_dir)

    save_dir = os.path.join(save_dir, "figures")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    make_graphs(save_dir, trained_results, test_results)

    # best_models, test_data = setup_visualize(args, save_dir, test_results)

    # visualize(save_dir, best_models, test_data)


if __name__ == "__main__":
    main()
