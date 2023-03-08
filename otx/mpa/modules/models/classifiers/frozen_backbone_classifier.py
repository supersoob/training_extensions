# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import torch
from mmcls.models.builder import CLASSIFIERS
from mmcls.models.classifiers.image import ImageClassifier

from .sam_classifier import SAMImageClassifier


@CLASSIFIERS.register_module()
class FrozenBackboneImageClassifier(ImageClassifier):
    @torch.no_grad()
    def extract_feat(self, img):
        return super().extract_feat(img)


@CLASSIFIERS.register_module()
class FrozenBackboneSAMImageClassifier(SAMImageClassifier):
    @torch.no_grad()
    def extract_feat(self, img):
        return super().extract_feat(img)
