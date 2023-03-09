# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.builder import CLASSIFIERS
from mmcls.models.builder import HEADS

from .sam_classifier import SAMImageClassifier
from otx.mpa.modules.models.heads.custom_cls_head import CustomLinearClsHead


class ResMLPBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        out += identity
        return out


@HEADS.register_module()
class ResMLPClassifier(CustomLinearClsHead):
    def __init__(self, num_classes, in_channels, *args, **kwargs) -> None:
        super().__init__(num_classes, in_channels, *args, **kwargs)
        self.in_channels = in_channels
        self.fc1 = nn.Linear(in_channels, 256)
        self.block1 = ResMLPBlock(256, 256)
        self.block2 = ResMLPBlock(256, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def fc(self, x):
        x = x.view(-1, self.in_channels)
        x = F.relu(self.fc1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.fc2(x)
        return x


@CLASSIFIERS.register_module()
class MMCLSVisionTransformerSAMImageClassifier(SAMImageClassifier):
    def extract_feat(self, img):
        x = super().extract_feat(img)
        if isinstance(x, tuple):
            x = x[-1]
        _, cls_token = x
        return cls_token


@CLASSIFIERS.register_module()
class MMCLSVisionTransformerwithCLIPWeightsSAMImageClassifier(SAMImageClassifier):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()


@CLASSIFIERS.register_module()
class CLIPVisionTransformerSAMImageClassifier(SAMImageClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.backbone = clip.load("ViT-L/14@336px", device)[0].visual

    @torch.no_grad()
    def extract_feat(self, img):
        return super().extract_feat(img)


@CLASSIFIERS.register_module()
class SAMImageClassifierTrainOnlyClassifier(SAMImageClassifier):
    def extract_feat(self, img):
        return img
