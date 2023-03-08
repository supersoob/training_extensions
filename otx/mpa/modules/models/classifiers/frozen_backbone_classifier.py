# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import torch
from mmcls.models.builder import CLASSIFIERS
from mmcls.models.classifiers.image import ImageClassifier

from .sam_classifier import SAMImageClassifier
import clip


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
    

@CLASSIFIERS.register_module()
class MMCLSVisionTransformerwithCLIPWeights(SAMImageClassifier):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()


@CLASSIFIERS.register_module()
class CLIPVisionTransformer(SAMImageClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.backbone = clip.load("ViT-L/14@336px", device)[0].visual

    @torch.no_grad()
    def extract_feat(self, img):
        return super().extract_feat(img)
