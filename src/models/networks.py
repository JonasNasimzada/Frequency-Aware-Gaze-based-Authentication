# This work is licensed under a "Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License"
# (https://creativecommons.org/licenses/by-nc-sa/4.0/).
#
# Author: Dillon Lohr (djl70@txstate.edu)
# Property of Texas State University.

from typing import List, Union

import numpy as np
import torch
import torch.nn as nn


# from ultralytics import YOLO


def init_weights(modules_list: Union[nn.Module, List[nn.Module]]) -> None:
    if not isinstance(modules_list, List):
        modules_list = [modules_list]

    for m in modules_list:
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0.0)


class Classifier(nn.Module):
    """Optional, pre-activation classification layer."""

    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes

        self.bn = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(input_dim, n_classes)

        init_weights(self.modules())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc(x)
        return x  # logits


class SimpleDenseNet(nn.Module):
    """
    Network with a single dense block.

    References
    ----------
    https://github.com/andreasveit/densenet-pytorch/blob/master/densenet.py
    """

    def __init__(
            self,
            depth: int,
            output_dim: int,
            growth_rate: int = 32,
            initial_channels: int = 2,
            max_dilation: int = 64,
            kernel_size: int = 3,
    ):
        super().__init__()
        # model = YOLO("yolo11n-cls.pt")
        n_fixed_layers = 1  # embedding layer
        n_layers_per_block = depth - n_fixed_layers
        assert n_layers_per_block > 0, "`depth` is too small"

        input_channels = initial_channels

        # Single dense block
        layers = [
            DenseBlock(
                n_layers_per_block,
                input_channels,
                growth_rate,
                max_dilation=max_dilation,
                skip_bn_relu_first_layer=True,
                kernel_size=kernel_size,
            )
        ]
        input_channels += n_layers_per_block * growth_rate
        self.block_sequence = nn.Sequential(*layers)

        # Global average pooling and embedding layer
        self.bn2 = nn.BatchNorm1d(input_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.output_dim = output_dim

        ### Transformers ###
        # from transformers import AutoImageProcessor, AutoModelForImageClassification
        # processor = AutoImageProcessor.from_pretrained("MattyB95/VIT-ASVspoof2019-Mel_Spectrogram-Synthetic-Voice-Detection")
        # self.model = AutoModelForImageClassification.from_pretrained("MattyB95/VIT-ASVspoof2019-Mel_Spectrogram-Synthetic-Voice-Detection")
        # self.fc = nn.Linear(768, output_dim)
        # self.model.classifier = self.fc

        # #### ResNet-50 ###
        # from torchvision.models import resnet50, ResNet50_Weights
        # self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # self.fc = nn.Linear(2048, output_dim)
        # self.model.fc  = self.fc

        #### ResNet-101 ###
        from torchvision.models import resnet101, ResNet101_Weights
        self.model = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.fc = nn.Linear(2048, output_dim)
        self.model.fc = self.fc

        # Initialize weights
        init_weights(self.modules())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print("HHeeHHrre:")
        print(x.shape)
        x = self.model(x)
        return x


class DenseBlock(nn.Module):
    """Series of convolution blocks with dense connections."""

    def __init__(
            self,
            n_layers: int,
            input_channels: int,
            growth_rate: int,
            max_dilation: int = 64,
            skip_bn_relu_first_layer: bool = True,
            kernel_size: int = 3,
    ):
        super().__init__()

        dilation_exp_mod = int(np.log2(max_dilation)) + 1

        def dilation_at_i(i: int) -> int:
            return 2 ** (i % dilation_exp_mod)

        layers = [
            ConvBlock(
                input_channels=input_channels + i * growth_rate,
                output_channels=growth_rate,
                dilation=dilation_at_i(i),
                skip_bn_relu=i == 0 and skip_bn_relu_first_layer,
                kernel_size=kernel_size,
            )
            for i in range(n_layers)
        ]
        self.block_sequence = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_sequence(x)
        return x


class ConvBlock(nn.Module):
    """BatchNorm1d + ReLU + Conv1d"""

    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            dilation: int = 1,
            skip_bn_relu: bool = False,
            kernel_size: int = 3,
    ):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_channels) if not skip_bn_relu else None
        self.relu = nn.ReLU(inplace=True) if not skip_bn_relu else None
        self.conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
            dilation=dilation,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        if self.bn is not None:
            out = self.bn(out)
        if self.relu is not None:
            out = self.relu(out)
        out = self.conv(out)
        return torch.cat([x, out], dim=1)
