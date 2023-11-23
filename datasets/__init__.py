# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

# from .coco import build as build_coco

# 定义一个函数，从数据集中获取COCO API
def get_coco_api_from_dataset(dataset):
    '''
    从数据集中获取COCO API
    Args:
        dataset: 数据集
    Returns:
        COCO 对象
    '''
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        # 如果数据集是torchvision.datasets.CocoDetection类型，则跳出循环
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        # 如果数据集是torch.utils.data.Subset类型，则将数据集设置为dataset.dataset
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    # 如果数据集是torchvision.datasets.CocoDetection类型，则返回dataset.coco
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
