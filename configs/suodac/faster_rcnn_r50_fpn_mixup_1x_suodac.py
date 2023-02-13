_base_='deepall_faster_rcnn_r50_fpn_1x_suodac.py'
# dataset settings
data_root = 'data/S-UODAC2020/'
dataset_type = 'SUODACDataset'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_scale = (640, 640)
train_pipeline = [
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', keep_ratio=True),
    dict(type='Pad', pad_to_square=True, pad_val=114.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

pre_train_pipeline = [
    dict(type='LoadImageFromSUODAC', to_float32=True, train=True, domain_file=data_root + 'VOC2007/ImageSets'),
    dict(type='LoadAnnotations', with_bbox=True)
]
train_dataset = dict(
    _delete_=True,
    type='MultiImageMixDataset',
    # dataset=dict(
    #         type=dataset_type,
    #         ann_file=data_root + 'VOC2007/ImageSets/type1.txt',
    #         img_prefix=data_root + "VOC2007/",
    #         pipeline=[
    #             dict(type='LoadImageFromFile', to_float32=True),
    #             dict(type='LoadAnnotations', with_bbox=True)
    #         ],
    #         filter_empty_gt=False),
    dataset=(
        dict(
            type=dataset_type,
            ann_file=data_root + 'VOC2007/ImageSets/type1.txt',
            img_prefix=data_root + "VOC2007/",
            pipeline=pre_train_pipeline,
            filter_empty_gt=False),
        dict(
            type=dataset_type,
            ann_file=data_root + 'VOC2007/ImageSets/type2.txt',
            img_prefix=data_root + "VOC2007/",
            pipeline=pre_train_pipeline,
            filter_empty_gt=False),
        dict(
            type=dataset_type,
            ann_file=data_root + 'VOC2007/ImageSets/type3.txt',
            img_prefix=data_root + "VOC2007/",
            pipeline=pre_train_pipeline,
            filter_empty_gt=False),
        dict(
            type=dataset_type,
            ann_file=data_root + 'VOC2007/ImageSets/type4.txt',
            img_prefix=data_root + "VOC2007/",
            pipeline=pre_train_pipeline,
            filter_empty_gt=False),
        dict(
            type=dataset_type,
            ann_file=data_root + 'VOC2007/ImageSets/type5.txt',
            img_prefix=data_root + "VOC2007/",
            pipeline=pre_train_pipeline,
            filter_empty_gt=False),
        dict(
            type=dataset_type,
            ann_file=data_root + 'VOC2007/ImageSets/type6.txt',
            img_prefix=data_root + "VOC2007/",
            pipeline=pre_train_pipeline,
            filter_empty_gt=False),
    ),
    pipeline=train_pipeline,
    dynamic_scale=img_scale)


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=train_dataset,)