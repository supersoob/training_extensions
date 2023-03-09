url = 'https://download.openmmlab.com/mmclassification/v0/vit/pretrain/vit-large-p16_3rdparty_pt-64xb64_in1k-224_20210928-0001f9a1.pth'
model = dict(
    type='MMCLSVisionTransformerSAMImageClassifier',
    backbone=dict(
        arch='large',
        drop_path_rate=0.0,
        drop_rate=0.0,
        final_norm=True,
        img_size=224,
        in_channels=3,
        init_cfg=dict(type='Pretrained', checkpoint=url, prefix='backbone'),
        interpolate_mode='bicubic',
        layer_cfgs=dict(),
        norm_cfg=dict(eps=1e-06, type='LN'),
        out_indices=-1,
        output_cls_token=True,
        patch_cfg=dict(),
        patch_size=16,
        qkv_bias=True,
        type='mmcls.VisionTransformer',
        with_cls_token=True),
    neck=None,
    head=dict(
        type='ResMLPClassifier',
        num_classes=2,
        in_channels=1024,
        loss=dict(type='FocalLoss', loss_weight=1.0)),
    task='classification')
fp16 = dict(loss_scale=512.0)
load_from = None
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)

evaluation = dict(metric=['accuracy', 'precision', 'recall', 'f1_score'])

optimizer = dict(
    _delete_=True,
    type='Adam',
    lr=0.001
)
