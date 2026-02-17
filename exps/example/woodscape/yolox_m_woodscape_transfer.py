#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
from yolox.exp import Exp as MyExp
"""

Experiment Transferlearning Pipeline for YOLOX_m on WoodScape Dataset only on Person class.
results in results/p1_reference_yolox_m

The dataset is prefiltered to only contain Person class annotations.
The pretrained model is the official yolox_m model trained on COCO with 80 classes.

        
"""


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        
        self.num_classes = 80 # default from pretrained model.
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.freeze_backbone = True # freeze backbone for transfer learning, only train head.
        
        # for training on HCI cluster with 48 GiB GPU(A40?), 
        # half precision training is recommended to save GPU memory and speed up training.
        self.fp16 = True
        self.batch_size = 16
        
        self.data_dir = "../dataset/woodscape_coco/"
        self.train_ann = "train.json"
        self.test_ann = "test.json"
        self.val_ann = "val.json"
        # self.image_size = (1963, 1216) # original image size, but we will resize to test_size for evaluation. 
        # --------------  training config --------------------- #
        # epoch number used for warmup
        self.warmup_epochs = 5
        # max training epoch
        self.max_epoch = 50
        # minimum learning rate during warmup
        self.warmup_lr = 0
        self.min_lr_ratio = 0.05
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 0.001 / 64.0 ## reduced lr for transfer learning, original is 0.01 / 64.0
        # name of LRScheduler
        self.scheduler = "yoloxwarmcos"
        # last #epoch to close augmention like mosaic
        self.no_aug_epochs = 15
        # apply EMA during training
        self.ema = True

        # weight decay of optimizer
        self.weight_decay = 5e-4
        # momentum of optimizer
        self.momentum = 0.9
        # log period in iter, for example,
        # if set to 1, user could see log every iteration.
        self.print_interval = 10
        # eval period in epoch, for example,
        # if set to 1, model will be evaluate after every epoch.
        self.eval_interval = 10 # todo maybe test every 5
        # save history checkpoint or not.
        # If set to False, yolox will only save latest and best ckpt.
        self.save_history_ckpt = True
        # name of experiment
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        # output image size during evaluation/test
        self.test_size = (640, 640)
        # confidence threshold during evaluation/test,
        # boxes whose scores are less than test_conf will be filtered
        self.test_conf = 0.01
        # nms threshold
        self.nmsthre = 0.65
   
    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        from yolox.data import COCODataset, TrainTransform

        dataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="images/train",
            img_size=self.test_size,
            preproc=TrainTransform()
            
        )
        return dataset

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        dataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.test_ann,
            name="images/test",
            img_size=self.test_size,
            preproc=ValTransform(legacy=kwargs.get("legacy", False)),
        )
        return dataset
