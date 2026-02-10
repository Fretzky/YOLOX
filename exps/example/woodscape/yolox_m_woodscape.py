#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
from yolox.exp import Exp as MyExp
"""

Experiment Reference Pipeline for YOLOX_m on WoodScape Dataset
results in results/p1_reference_yolox_m
        
"""


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.test_size = (640, 640)
        self.num_classes = 80 # default from pretrained model.
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
         
        self.nmsthre = 0.65 # default from pretrained model
   
        # Define yourself dataset path
        self.data_dir = os.environ.get("WOODSCAPE_COCO_DIR", "/app/data/woodscape_coco2")
        self.train_ann = "train.json"
        self.val_ann = "val.json"
   

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        dataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="images/val",
            img_size=self.test_size,
            preproc=ValTransform(legacy=kwargs.get("legacy", False)),
        )
        return dataset
