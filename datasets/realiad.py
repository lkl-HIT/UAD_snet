import os
from enum import Enum

import numpy as np
import PIL
from PIL import Image
import torch
import json
from torchvision import transforms

#内存监测
import psutil
import sys

_CLASSNAMES = [
    'audiojack', 
    'bottle_cap', 
    'button_battery', 
    'end_cap', 
    'eraser', 
    'fire_hood', 
    'mint', 
    'mounts', 
    'pcb', 
    'phone_battery', 
    'plastic_nut', 
    'plastic_plug', 
    'porcelain_doll', 
    'regulator', 
    'rolled_strip_base', 
    'sim_card_set', 
    'switch', 
    'tape', 
    'terminalblock', 
    'toothbrush', 
    'toy', 
    'toy_brick', 
    'transistor1', 
    'usb', 
    'usb_adaptor', 
    'u_block', 
    'vcpill', 
    'wooden_beads', 
    'woodstick', 
    'zipper',
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]



def log_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"[{tag}] Memory usage: {mem_info.rss / 1024 ** 2:.2f} MB")


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class RealIADDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        rotate_degrees=0,
        translate=0,
        brightness_factor=0,
        contrast_factor=0,
        saturation_factor=0,
        gray_p=0,
        h_flip_p=0,
        v_flip_p=0,
        scale=0,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split
        self.transform_std = IMAGENET_STD
        self.transform_mean = IMAGENET_MEAN
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()
        del self.imgpaths_per_class

        self.transform_img = [
            transforms.Resize(resize),
            # transforms.RandomRotation(rotate_degrees, transforms.InterpolationMode.BILINEAR),
            # transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),#数据增强->随机亮度，对比度，饱和度变换
            # transforms.RandomHorizontalFlip(h_flip_p),#数据增强->随机水平翻转
            # transforms.RandomVerticalFlip(v_flip_p),#数据增强->随机垂直翻转
            # transforms.RandomGrayscale(gray_p),#数据增强->随机灰度
            # transforms.RandomAffine(rotate_degrees, 
            #                         translate=(translate, translate),
            #                         scale=(1.0-scale, 1.0+scale),
            #                         interpolation=transforms.InterpolationMode.BILINEAR),#数据增强->随机仿射变换
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "OK"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        
        imgpaths_per_class = {}
        data_to_iterate = []

        for classname in self.classnames_to_use:
            json_path = os.path.join(self.source, 'realiad_jsons', 'realiad_jsons', classname + '.json')

            with open(json_path) as file:
                class_json = file.read()
            class_json = json.loads(class_json)

            imgpaths_per_class[classname] = class_json[self.split.value]#no val

            for sample in imgpaths_per_class[classname]:
                image_path = os.path.join(self.source, 'realiad_1024', classname, sample['image_path'])
                data_tuple = [classname, sample['anomaly_class'], image_path]
                label = sample['anomaly_class'] != 'OK'
                if label:
                    gt_paths = os.path.join(self.source, 'realiad_1024', classname, sample['mask_path'])
                    data_tuple.append(gt_paths)
                else:
                    data_tuple.append(None)
                data_to_iterate.append(data_tuple)

        # Unrolls the data dictionary to an easy-to-iterate list.
        

        return imgpaths_per_class, data_to_iterate




class RealIADDataset_1(torch.utils.data.Dataset):
    def __init__(
            self, 
            root, 
            classname, 
            resize=256,
            imagesize=224,
            split=DatasetSplit.TRAIN,
            train_val_split=1.0,
            rotate_degrees=0,
            translate=0,
            brightness_factor=0,
            contrast_factor=0,
            saturation_factor=0,
            gray_p=0,
            h_flip_p=0,
            v_flip_p=0,
            scale=0,
            **kwargs,
    ):
        self.img_path = os.path.join(root, 'realiad_1024', classname)
        self.img_class = classname #暂不支持多类读取
        
        data_transforms = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.CenterCrop(imagesize),
            transforms.Normalize(mean=IMAGENET_MEAN,
                                std=IMAGENET_STD)])
        gt_transforms = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor()])
        
        
        self.transform = data_transforms
        self.gt_transform = gt_transforms
        self.phase = split.value
        self.imagesize = (3, imagesize, imagesize)

        json_path = os.path.join(root, 'realiad_jsons', 'realiad_jsons', classname + '.json')
        with open(json_path) as file:
            class_json = file.read()
        class_json = json.loads(class_json)

        # self.img_paths, self.gt_paths, self.labels, self.types,self.img_class = [], [], [], [], []
        self.img_paths, self.gt_paths, self.labels, self.types = [], [], [], []

        data_set = class_json[self.phase]
        for sample in data_set:
            self.img_paths.append(os.path.join(root, 'realiad_1024', classname, sample['image_path']))
            label = sample['anomaly_class'] != 'OK'
            if label:
                self.gt_paths.append(os.path.join(root, 'realiad_1024', classname, sample['mask_path']))
            else:
                self.gt_paths.append(None)
            self.labels.append(label)
            self.types.append(sample['anomaly_class'])

        self.img_paths = np.array(self.img_paths)
        self.gt_paths = np.array(self.gt_paths)
        self.labels = np.array(self.labels)
        self.types = np.array(self.types)
        self.cls_idx = 0

        

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        # if self.phase == 'train':
        #     return img, label

        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return {
            'image': img,
            'mask': gt,
            'classname': self.img_class ,
            'anomaly': img_type,
            'is_anomaly': int(label != 0),
            'image_name': "/".join(img_path.split("/")[-4:]),
            'image_path': img_path,
        }