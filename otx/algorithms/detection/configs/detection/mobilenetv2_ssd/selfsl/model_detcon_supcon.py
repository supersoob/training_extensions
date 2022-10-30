"""Model configuration of Self-SL with ATSS model for Detection Task."""

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

# pylint: disable=invalid-name

_base_ = [
    "../model.py",
]

model = dict(
    type='DetConBSupCon',
    pretrained='https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/object_detection/v2/mobilenet_v2-2s_ssd-992x736.pth',
    num_classes=None, # to be set based on dataset
    num_samples=16,
    downsample=16,
    input_transform='resize_concat',
    in_index=[0,1],
    projector=dict(
        in_channels=416,
        hid_channels=832,
        out_channels=256,
        norm_cfg=dict(type='BN1d', requires_grad=True),
        with_avg_pool=False
    ),
    predictor=dict(
        in_channels=256,
        hid_channels=832,
        out_channels=256,
        norm_cfg=dict(type='BN1d', requires_grad=True),
        with_avg_pool=False
    ),
    detcon_loss_cfg=dict(type='DetConBLoss', temperature=0.1),
)

load_from = None

custom_hooks = [
    dict(
        type='LazyEarlyStoppingHook',
        start=3,
        patience=10,
        iteration_patience=0,
        metric='mAP',
        interval=1,
        priority=75,
    ),
    dict(
        type='SwitchPipelineHook',
        interval=1
    )
]
