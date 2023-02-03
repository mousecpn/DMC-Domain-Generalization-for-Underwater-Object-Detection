# Achieving Domain Generalization in Underwater Object Detection by Domain Mixup and Contrastive Learning

This repository contains the code (in PyTorch) for the paper:

[Achieving Domain Generalization in Underwater Object Detection by Domain Mixup and Contrastive Learning](https://arxiv.org/abs/2104.02230)

## Introduction

The performance of existing underwater object detection methods degrades seriously when facing domain shift caused by complicated underwater environments. Due to the limitation of the number of domains in the dataset, deep detectors just easily memorize a few seen domain, which leads to low generalization ability. There are two common ideas to improve the domain generalization performance. First, it can be inferred that the detector trained on as many domains as possible is domain-invariant. Second, for the images with same semantic content in different domains, its hidden features should be equivalent. This paper further excavate these two ideas and proposes a domain generalization framework (named DMC) that learns how to generalize across domains from Domain Mixup and Contrastive Learning. First, based on the formation of underwater images, an image in an underwater environment is the linear transformation of another underwater environment. Thus, a style transfer model, which outputs a linear transformation matrix instead of the whole image, is proposed to transform images from one source domain to another, enriching the domain diversity of the training data. Second, mixup operation interpolates different domains on feature level, sampling new domains on the domain manifold. Third, contrastive loss is selectively applied on features from different domain to force model to learn domain invariant features but retain the discriminative capacity. With our method, detectors will be robust to domain shift. Also, a domain generalization benchmark S-UODAC2020 for detection is set up to measure the performance of our method. Comprehensive experiments on S-UODAC2020 and two object recognition benchmarks (PACS and VLCS) demonstrate that the proposed method is able to learn domain-invariant representations, and outperforms other domain generalization methods.

![image](https://user-images.githubusercontent.com/46233799/175855556-5bf4701a-2e11-4c98-9053-97dcf89bec95.png)


## Dependencies

- Python==3.7.6
- PyTorch==1.0
- mmdetection==2.17.0
- mmcv==1.3.8
- numpy==1.16.3

## Installation

The basic installation follows with [mmdetection](https://github.com/mousecpn/mmdetection/blob/master/docs/get_started.md). It is recommended to use manual installation. 

## **Download Datasets**

S-UODAC2020: https://drive.google.com/open?id=1mAGqjP-N6d-FRMy5I8sDkMRCuZuD1Na7&authuser=pinhaosong%40gmail.com&usp=drive_fs

After downloading all datasets, create S-UODAC2020 document.

```
$ cd data
$ mkdir S-UODAC2020
```

It is recommended to symlink the dataset root to `$data/S-UODAC2020`.

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

## Train

```
$ python tools/train.py configs/suodac/dmc_faster_rcnn_r50_fpn_1x_suodac.py
```

## Test

```
$ python tools/test.py configs/suodac/dmc_faster_rcnn_r50_fpn_1x_suodac.py <path/to/checkpoints>
```

This repo also provides the re-implementation of JiGEN, MMD-AAE, CCSA, CIDDG, DANN, CrossGrad in Faster R-CNN framework.

### Citation

```
@article{chen2023achieving,
  title={Achieving domain generalization for underwater object detection by domain mixup and contrastive learning},
  author={Chen, Yang and Song, Pinhao and Liu, Hong and Dai, Linhui and Zhang, Xiaochuan and Ding, Runwei and Li, Shengquan},
  journal={Neurocomputing},
  year={2023},
  publisher={Elsevier}
}
```

