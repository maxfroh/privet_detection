#
# Fast R-CNN model
#

# cd ..
# python privet_detection/src/models/fast_rcnn.py

import sys
import logging
from typing import Callable

import torch
from torch.nn import Module, Conv2d
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.resnet import ResNet, resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models._utils import IntermediateLayerGetter
from torch.utils.data import DataLoader

out_channels = {
    resnet18: 512,
    resnet34: 512,
    resnet50: 2048,
    resnet101: 2048,
    resnet152: 2048
}


def FasterRCNNResNet101(classes: list[str] = ["privet", "yew", "path", "background"], backbone_model: Callable[..., ResNet] = resnet101, num_channels: int = 3):
    """
    """
    num_classes = len(classes)
    
    resnet = backbone_model(weights=None)

    # replacing roi head predictor
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # multichannel weights
    if num_channels != 3:
        old_resnet_conv1_weights = resnet.conv1.weight
        resnet.conv1 = Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        with torch.no_grad():
            resnet.conv1.weight[:, :3] = old_resnet_conv1_weights
            # resnet.conv1.weight[:, 3:]

    return_layers = {"layer4": "0"}
    backbone = IntermediateLayerGetter(resnet, return_layers=return_layers)
    # must specify out_channels size
    backbone.out_channels = out_channels[backbone_model]

    # making anchor generator
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator,
                      box_roi_pool=roi_pooler)
    
    # multichannel transform
    if num_channels != 3:
        image_mean = model.transform.image_mean
        image_mean.extend([0.5] * (num_channels - len(image_mean)))
        image_std = model.transform.image_std
        image_std.extend([0.5] * (num_channels - len(image_std)))
        grcnn_trans = GeneralizedRCNNTransform(
            min_size=model.transform.min_size, max_size=model.transform.max_size, image_mean=image_mean, image_std=image_std)
        model.transform = grcnn_trans

    return model


def main():
    print("Error")


if __name__ == "__main__":
    main()
