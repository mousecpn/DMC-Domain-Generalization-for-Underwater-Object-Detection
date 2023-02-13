_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

model = dict(
    type='MMDAAEFasterRCNN',
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='CIDDGRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='CCSAConvFCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=4,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))))

dataset_type = 'SUODACDataset'
classes = ('echinus','starfish','holothurian','scallop')
data_root = 'data/S-UODAC2020/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromSUODAC', to_float32=True, train=True, domain_file=data_root+'VOC2007/ImageSets'),
    # dict(type='GeneratePuzzle',img_norm_cfg=img_norm_cfg,jig_classes = 30),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels','domain_label']),
]
test_pipeline = [
    dict(type='LoadImageFromSUODAC', to_float32=True, train=True, domain_file=data_root+'VOC2007/ImageSets'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


data = dict(
    samples_per_gpu = 2,
    workers_per_gpu = 4,
    train = (
            dict(
                type=dataset_type,
                ann_file=data_root + 'VOC2007/ImageSets/type1.txt',
                img_prefix=data_root + "VOC2007/",
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                ann_file=data_root + 'VOC2007/ImageSets/type2.txt',
                img_prefix=data_root + "VOC2007/",
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                ann_file=data_root + 'VOC2007/ImageSets/type3.txt',
                img_prefix=data_root + "VOC2007/",
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                ann_file=data_root + 'VOC2007/ImageSets/type4.txt',
                img_prefix=data_root + "VOC2007/",
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                ann_file=data_root + 'VOC2007/ImageSets/type5.txt',
                img_prefix=data_root + "VOC2007/",
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                ann_file=data_root + 'VOC2007/ImageSets/type6.txt',
                img_prefix=data_root + "VOC2007/",
                pipeline=train_pipeline),
    ),
    val=dict(
            type=dataset_type,
            ann_file=data_root + 'VOC2007/ImageSets/type7.txt',
            img_prefix=data_root + "VOC2007/",
            pipeline=test_pipeline),
    test=dict(
            type=dataset_type,
            ann_file=data_root + 'VOC2007/ImageSets/type7.txt',
            img_prefix=data_root + "VOC2007/",
            pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')