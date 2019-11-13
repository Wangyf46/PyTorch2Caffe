#!/usr/bin/python
#_*_ coding:UTF-8 _*_
#pytorch version = 1.0.0
import os
import re
import importlib
import configparser
from torchvision.transforms import functional as F
from torch.autograd import Variable
import torch
import sys
import cv2
sys.path.append('.')

cf = configparser.ConfigParser()

class PreProcess(object):
    def __init__(self,config_path):
        # self.model_class_dir = model_class_dir
        # config_path = model_class_dir+"/model.conf"

        cf = configparser.ConfigParser()
        cf.read(config_path)#read configuration file
        self.module_name = cf.get("pytorch", "module_name")
        self.test_image = cf.get("pytorch","test_image")
        self.model_transform_result_dir="trans_result/"+self.module_name
        self.class_name = cf.get("pytorch", "class_name")
        mean_val_str = cf.get("pytorch", "mean_val")
        self.mean_value=self.string2list_float(mean_val_str)
        std_val_str = cf.get("pytorch","std")
        self.std=self.string2list_float(std_val_str)
        self.color = cf.get("pytorch","color")
        input_size_str = cf.get("pytorch","input_size")
        self.input_size = self.string2list_int(input_size_str)
        self.cf = cf

    def string2list_int(self,list_string):
        list_string=list_string.split(",")
        list_int =[]
        for item in list_string:
            list_int.append(int(item))
        return list_int

    def string2list_float(self,list_string):
        list_string=list_string.split(",")
        list_float =[]
        for item in list_string:
            list_float.append(float(item))
        return list_float

    def processImage(self,img):
        img_c, img_h, img_w = self.input_size
        img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        if self.color=="RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2读出的格式是BGR,需要转换为RGB
        img = img.transpose((2, 0, 1))  # opencv读出的格式是HWC,需要转换为CHW
        img = torch.from_numpy(img).float()
        img = F.normalize(img, self.mean_value, self.std)
        img = img.expand(1, img_c, img_h, img_w)  # 送入模型的应该是B*C×H*W
        return img
