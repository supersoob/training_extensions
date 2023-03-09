data = dict(
    samples_per_gpu=2048,
    workers_per_gpu=2,
    train=dict(
        type='AIEDataset',
        pipeline=[
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ],
        features='otx-workspace-AIE/revised_pickles/train_split_clip_ViT-L-p-14-336_feature.pickle'),
    val=dict(
        type='AIEDataset',
        pipeline=[
            dict(type='Collect', keys=['img'])
        ],
        features='otx-workspace-AIE/revised_pickles/test_split_clip_ViT-L-p-14-336_feature.pickle'),
    test=dict(
        type='AIEDataset',
        pipeline=[
            dict(type='Collect', keys=['img'])
        ],
        features='otx-workspace-AIE/revised_pickles/test_split_clip_ViT-L-p-14-336_feature.pickle'),
)
