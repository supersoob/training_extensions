"""Data Pipeline of SSD model for Detection Task."""

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

dataset_type = "CocoDataset"
img_size = (864, 864)
img_norm_cfg = dict(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)

train_pipeline = [
    dict(type="LoadImageFromFile"), # delete to_float32
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="SelfSLCompose",
        pipeline1=[
            dict(type="MinIoURandomCrop", min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.1),
            dict(type="Resize", img_scale=img_size, keep_ratio=False),
            dict(type="RandomFlip", flip_ratio=0.5),
            dict(type="ProbCompose",
                transforms=[
                    dict(
                        type="SelfSLColorJitter",
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.2,
                        hue=0.1
                    )
                ], probs=[0.8]),
            dict(type="SelfSLRandomGrayscale", p=0.2),
            dict(type="GaussianBlur", kernel_size=23),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
        ],
        pipeline2=[
            dict(type="MinIoURandomCrop", min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.1),
            dict(type="Resize", img_scale=img_size, keep_ratio=False),
            dict(type="RandomFlip", flip_ratio=0.5),
            dict(type="ProbCompose",
                transforms=[
                    dict(
                        type="SelfSLColorJitter",
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.2,
                        hue=0.1
                    )
                ], probs=[0.8]),
            dict(type="SelfSLRandomGrayscale", p=0.2),
            dict(type="ProbCompose", transforms=[dict(type="GaussianBlur", kernel_size=23)], probs=[0.1]),
            dict(type="ProbCompose", transforms=[dict(type="Solarization", threshold=128)], probs=[0.2]),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
        ],
    ),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_size,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=False),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=10,
    workers_per_gpu=4,
    train=dict(
        type="RepeatDataset",
        times=1,
        adaptive_repeat_times=True,
        dataset=dict(
            type=dataset_type,
            ann_file="data/coco/annotations/instances_train2017.json",
            img_prefix="data/coco/train2017",
            pipeline=train_pipeline,
        ),
    ),
    val=dict(
        type=dataset_type,
        ann_file="data/coco/annotations/instances_val2017.json",
        img_prefix="data/coco/val2017",
        test_mode=True,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file="data/coco/annotations/instances_val2017.json",
        img_prefix="data/coco/val2017",
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
