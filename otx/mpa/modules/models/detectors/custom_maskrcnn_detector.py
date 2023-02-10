# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import functools

import torch
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.mask_rcnn import MaskRCNN

from otx.mpa.deploy.utils import is_mmdeploy_enabled
from otx.mpa.modules.utils.task_adapt import map_class_names
from otx.mpa.utils.logger import get_logger

from .l2sp_detector_mixin import L2SPDetectorMixin
from .sam_detector_mixin import SAMDetectorMixin

from torch import nn
import numpy as np

logger = get_logger()


class TileClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(256 * 6 * 6, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 1)
        )
        self.loss_fun = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0]))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, img):
        x = self.features(img)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.classifier(x)
        return y

    def loss(self, pred, target):
        loss = self.loss_fun(pred, target)
        return loss

    def simple_test(self, img):
        return self.sigmoid(self.forward(img))[0][0]


@DETECTORS.register_module()
class CustomMaskRCNN(SAMDetectorMixin, L2SPDetectorMixin, MaskRCNN):
    def __init__(self, *args, task_adapt=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tile_classifier = TileClassifier()

        # Hook for class-sensitive weight loading
        if task_adapt:
            self._register_load_state_dict_pre_hook(
                functools.partial(
                    self.load_state_dict_pre_hook,
                    self,  # model
                    task_adapt["dst_classes"],  # model_classes
                    task_adapt["src_classes"],  # chkpt_classes
                )
            )

    @staticmethod
    def load_state_dict_pre_hook(model, model_classes, chkpt_classes, chkpt_dict, prefix, *args, **kwargs):
        """Modify input state_dict according to class name matching before weight loading"""
        logger.info(f"----------------- CustomMaskRCNN.load_state_dict_pre_hook() called w/ prefix: {prefix}")

        # Dst to src mapping index
        model_dict = model.state_dict()
        model_classes = list(model_classes)
        chkpt_classes = list(chkpt_classes)
        model2chkpt = map_class_names(model_classes, chkpt_classes)
        logger.info(f"{chkpt_classes} -> {model_classes} ({model2chkpt})")

        # List of class-relevant params & their row-stride
        param_strides = {
            "roi_head.bbox_head.fc_cls.weight": 1,
            "roi_head.bbox_head.fc_cls.bias": 1,
            "roi_head.bbox_head.fc_reg.weight": 4,  # 4 rows (bbox) for each class
            "roi_head.bbox_head.fc_reg.bias": 4,
        }

        for model_name, stride in param_strides.items():
            chkpt_name = prefix + model_name
            if model_name not in model_dict or chkpt_name not in chkpt_dict:
                logger.info(f"Skipping weight copy: {chkpt_name}")
                continue

            # Mix weights
            model_param = model_dict[model_name].clone()
            chkpt_param = chkpt_dict[chkpt_name]
            for m, c in enumerate(model2chkpt):
                if c >= 0:
                    # Copying only matched weight rows
                    model_param[(m) * stride : (m + 1) * stride].copy_(chkpt_param[(c) * stride : (c + 1) * stride])
            if model_param.shape[0] > len(model_classes * stride):  # BG class
                c = len(chkpt_classes)
                m = len(model_classes)
                model_param[(m) * stride : (m + 1) * stride].copy_(chkpt_param[(c) * stride : (c + 1) * stride])

            # Replace checkpoint weight by mixed weights
            chkpt_dict[chkpt_name] = model_param

    def forward_train(
            self,
            img,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore=None,
            gt_masks=None,
            proposals=None,
            **kwargs):
        losses = dict()
        targets = [len(gt_bbox) > 0 for gt_bbox in gt_bboxes]
        pred = self.tile_classifier(img)
        target_labels = torch.tensor(targets, device=pred.device)
        loss_tile = self.tile_classifier.loss(pred, target_labels.unsqueeze(1).float())

        losses.update(dict(loss_tile=loss_tile))

        if not any(targets):
            return losses

        img = img[targets]
        img_metas = [item for keep, item in zip(targets, img_metas) if keep]
        gt_labels = [item for keep, item in zip(targets, gt_labels) if keep]
        gt_bboxes = [item for keep, item in zip(targets, gt_bboxes) if keep]
        gt_masks = [item for keep, item in zip(targets, gt_masks) if keep]
        if gt_bboxes_ignore:
            gt_bboxes_ignore = [
                item for keep, item in zip(targets, gt_bboxes_ignore) if keep]
        rcnn_loss = super().forward_train(
                        img,
                        img_metas,
                        gt_bboxes,
                        gt_labels,
                        gt_bboxes_ignore, gt_masks, proposals, **kwargs)
        losses.update(rcnn_loss)
        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        keep = self.tile_classifier.simple_test(img) > 0.45

        if not keep:
            tmp_results = []
            num_classes = 1
            bbox_results = []
            mask_results = []
            for _ in range(num_classes):
                bbox_results.append(np.empty((0, 5), dtype=np.float32))
                mask_results.append([])
            tmp_results.append((bbox_results, mask_results))
            return tmp_results

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)


if is_mmdeploy_enabled():
    from mmdeploy.core import FUNCTION_REWRITER, mark
    from mmdeploy.utils import is_dynamic_shape

    from otx.mpa.modules.hooks.recording_forward_hooks import (
        ActivationMapHook,
        FeatureVectorHook,
    )

    @mark("tile_classifier", inputs=["img"], outputs=["prob"])
    def tile_classifier__simple_test_impl(ctx, self, img):
        """ Tile Classifier Simple Test Impl with added mmdeploy marking for model partitioning

        Args:
            ctx (_type_): _description_
            img (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.sigmoid(self.forward(img))[0][0]

    @FUNCTION_REWRITER.register_rewriter(
        "otx.mpa.modules.models.detectors.custom_maskrcnn_detector.TileClassifier.simple_test"
    )
    def tile_classifier__simple_test(ctx, self, img, **kwargs):
        """ Tile Classifier Simple Test Rewriter

        Args:
            ctx (_type_): 
            img (_type_): 

        Returns:
            _type_: _description_
        """
        return tile_classifier__simple_test_impl(ctx, self, img)


    @mark("custom_maskrcnn_forward", inputs=["input"], outputs=["dets", "labels", "masks", "feats", "saliencies", "tile_prob"])
    def __forward_impl(ctx, self, img, img_metas, **kwargs):
        assert isinstance(img, torch.Tensor)

        deploy_cfg = ctx.cfg
        is_dynamic_flag = is_dynamic_shape(deploy_cfg)
        # get origin input shape as tensor to support onnx dynamic shape
        img_shape = torch._shape_as_tensor(img)[2:]
        if not is_dynamic_flag:
            img_shape = [int(val) for val in img_shape]
        img_metas[0]["img_shape"] = img_shape
        return self.simple_test(img, img_metas, **kwargs)

    @FUNCTION_REWRITER.register_rewriter(
        "otx.mpa.modules.models.detectors.custom_maskrcnn_detector.CustomMaskRCNN.forward"
    )
    def custom_maskrcnn__forward(ctx, self, img, img_metas=None, return_loss=False, **kwargs):
        if img_metas is None:
            img_metas = [{}]
        else:
            assert len(img_metas) == 1, "do not support aug_test"
            img_metas = img_metas[0]

        if isinstance(img, list):
            img = img[0]

        return __forward_impl(ctx, self, img, img_metas=img_metas, **kwargs)

    @FUNCTION_REWRITER.register_rewriter(
        "otx.mpa.modules.models.detectors.custom_maskrcnn_detector.CustomMaskRCNN.simple_test"
    )
    def custom_mask_rcnn__simple_test(ctx, self, img, img_metas, proposals=None, **kwargs):
        assert self.with_bbox, "Bbox head must be implemented."
        tile_prob = self.tile_classifier.simple_test(img)

        x = self.extract_feat(img)
        feature_vector = FeatureVectorHook.func(x)
        saliency_map = ActivationMapHook.func(x[-1])
        if proposals is None:
            proposals, _ = self.rpn_head.simple_test_rpn(x, img_metas)
        out = self.roi_head.simple_test(x, proposals, img_metas, rescale=False)
        # NOTE: tile_prob should not returned but it is required for model tracing
        return (*out, feature_vector, saliency_map, tile_prob)
