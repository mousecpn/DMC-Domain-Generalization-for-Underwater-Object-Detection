This repository contains the code (in PyTorch) for the paper:

[Achieving Domain Generalization in Underwater Object Detection by Domain Mixup and Contrastive Learning](https://arxiv.org/abs/2104.02230)

### Dependencies

- Python==3.7.6
- PyTorch==1.0
- mmdetection==2.17.0
- mmcv==1.3.8
- numpy==1.16.3

### Installation

The basic installation follows with [mmdetection](https://github.com/mousecpn/mmdetection/blob/master/docs/get_started.md). It is recommended to use manual installation. 

**Download Datasets**

S-UODAC2020: https://drive.google.com/open?id=1mAGqjP-N6d-FRMy5I8sDkMRCuZuD1Na7&authuser=pinhaosong%40gmail.com&usp=drive_fs

After downloading all datasets, create S-UODAC2020 document.

```
$ cd data
$ mkdir S-UODAC2020
```

It is recommended to symlink the dataset root to `$DG-YOLO/data/URPC2019`.

```
DMC-Domain-Generalization-for-Underwater-Object-Detection
├── data
│   ├── S-UODAC2020
│   │   ├── Annotations
│   │   ├── COCO_Annotations
│   │   ├── type1
│   │   ├── type2
│   │   ├── type3
│   │   ├── type4
│   │   ├── type5
│   │   ├── type7
│   │   ├── VOC2007
```

### Train

```
$ python tools/train.py configs/suodac/dmc_faster_rcnn_r50_fpn_1x_suodac.py
```

### Test

```
$ python tools/test.py configs/suodac/dmc_faster_rcnn_r50_fpn_1x_suodac.py <path/to/checkpoints>
```

This repo also provides the re-implementation of JiGEN, MMD-AAE, CCSA, CIDDG, DANN, CrossGrad in Faster R-CNN framework.

### Citation

```
@article{song2021achieving,
  title={Achieving Domain Generalization in Underwater Object Detection by Image Stylization and Domain Mixup},
  author={Song, Pinhao and Dai, Linhui and Yuan, Peipei and Liu, Hong and Ding, Runwei},
  journal={arXiv preprint arXiv:2104.02230},
  year={2021}
}
```