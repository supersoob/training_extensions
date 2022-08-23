_base_ = [
    './coco_data_pipeline.py'
]

model = dict(
    type='SOLOv2',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5),
    mask_head=dict(
        type='SOLOV2Head',
        num_classes=80,
        in_channels=256,
        feat_channels=256,
        stacked_convs=2,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 56), (28, 112), (56, 224), (112, 448), (224, 896)),
        pos_scale=0.2,
        num_grids=[40, 36, 24, 16, 12],
        cls_down_index=0,
        mask_feature_head=dict(
            feat_channels=128,
            start_level=0,
            end_level=3,
            out_channels=128,
            mask_stride=4,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
        loss_mask=dict(type='DiceLoss', use_sigmoid=True, loss_weight=3.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0)),
    test_cfg=dict(
        nms_pre=500,
        score_thr=0.1,
        mask_thr=0.5,
        filter_thr=0.05,
        kernel='gaussian',
        sigma=2.0,
        max_per_img=100))


cudnn_benchmark = True
evaluation = dict(interval=1, metric='mAP', save_best='mAP', iou_thr=[0.5])
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='ReduceLROnPlateau',
    metric='mAP',
    patience=5,
    iteration_patience=300,
    interval=1,
    min_lr=0.000001,
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1.0 / 3)

runner = dict(type='EpochRunnerWithCancel', max_epochs=300)
checkpoint_config = dict(interval=5)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'output'
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_light_r50_fpn_3x_coco/solov2_light_r50_fpn_3x_coco_20220512_165256-c93a6074.pth'
resume_from = None
workflow = [('train', 1)]

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        patience=10,
        iteration_patience=0,
        metric='mAP',
        interval=1,
        priority=75)
]