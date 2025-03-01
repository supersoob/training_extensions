"""EfficientNet-V2 for multi-class config."""

# pylint: disable=invalid-name

_base_ = ["../../../../recipes/stages/classification/incremental.yaml", "../base/models/efficientnet_v2.py"]

model = dict(
    type="SAMImageClassifier",
    task="classification",
    backbone=dict(
        version="s_21k",
    ),
    head=dict(type="CustomLinearClsHead", loss=dict(type="CrossEntropyLoss", loss_weight=1.0)),
)

fp16 = dict(loss_scale=512.0)
