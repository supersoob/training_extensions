dataset_type = 'CocoDataset'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
    dict(type='Resize',
         img_scale=[(768, 512), (768, 480), (768, 448), (768, 416), (768, 384), (768, 352)],
         multiscale_mode='value',
         keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(448, 768),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(samples_per_gpu=4,
            workers_per_gpu=2,
            train=dict(
                type='RepeatDataset',
                adaptive_repeat_times=True,
                times=1,
                dataset=dict(
                    type=dataset_type,
                    ann_file='data/coco/annotations/instances_train2017.json',
                    img_prefix='data/coco/train2017',
                    pipeline=train_pipeline)),
            val=dict(
                type=dataset_type,
                test_mode=True,
                ann_file='data/coco/annotations/instances_val2017.json',
                img_prefix='data/coco/val2017',
                pipeline=test_pipeline),
            test=dict(
                type=dataset_type,
                test_mode=True,
                ann_file='data/coco/annotations/instances_val2017.json',
                img_prefix='data/coco/val2017',
                pipeline=test_pipeline))