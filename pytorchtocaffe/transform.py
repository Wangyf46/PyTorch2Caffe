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
from pre_process import PreProcess
import argparse
sys.path.append('.')
import pdb


class Transform(PreProcess):
    def __init__(self,model_class_dir):
        self.model_class_dir = model_class_dir
        config_path = model_class_dir+"/model.conf"
        super(Transform, self).__init__(config_path)
    def createDir(self,path):
        if not os.path.exists(path):
            os.makedirs(path)

    def save_result(self,result_list,file_name):
        f=open(file_name, 'w')
        # result=result_list.detach().numpy()
        # for index in range(result.shape[1]):
        #     f.write(str(result[0,index])+"\n")
        for item in result_list:
            f.write(str(item))
        f.close()

    def saveCaffe2tensorRTParam(self):
        self.cf.add_section('caffe')#write configuration file
        self.cf.set('caffe', 'prototxt', self.caffe_prototxt)
        self.cf.set('caffe', 'caffemodel', self.caffe_model)
        self.cf.set('caffe', 'name',self.name)
        self.cf.set('caffe', 'input_layer_name', "blob1")
        output_string=self.output_blobs[0]
        for index in range(1,len(self.output_blobs)):
            output_string = output_string+","+self.output_blobs[index]
        self.cf.set('caffe', 'output_layer_name',output_string)
        output_name = self.output_names[0]
        for index in range(1, len(self.output_names)):
            output_name = output_name + "," + self.output_names[index]
        self.cf.set('caffe', 'output_top_name', output_name)
        self.cf.set('caffe', 'engine_model',self.engine_model)
        self.cf.set('caffe', 'model_name', "other")
        save_path = self.model_transform_result_dir+"/caffe2tensorRT.param"
        self.cf.write(open(save_path,'w'))
        pass

    def transform2caffe(self):
        module_name = self.module_name
        trnsform_class = self.class_name
        sys.path.append(self.model_class_dir)
        print(self.model_class_dir)
        ip_module = __import__(module_name)
        trans_class = getattr(ip_module, trnsform_class)
        # from FssdMobilev1Net_temp import FssdMobilev1Net
        # model = FssdMobilev1Net()
        model = trans_class()
        input_size  = [1, self.input_size[0], self.input_size[1], self.input_size[2]]
        input_variable = Variable(torch.ones(input_size))
        #print(input_variable)
        
        #run test image and save result
        img = cv2.imread(self.test_image)
        output_result = model.run(img)
        #print(output_result)

        model_transform_result_dir = self.model_transform_result_dir
        self.createDir(self.model_transform_result_dir)
        self.save_result(output_result,self.model_transform_result_dir+"/pytorch.result")

        #generate caffemodel
        pytorch_to_caffe_dir = "./pytorch2caffe"
        sys.path.append(pytorch_to_caffe_dir)
        trans_module = 'pytorch_to_caffe'
        self.name = module_name + '_model'
        pytorch_to_caffe = __import__(trans_module)
        pytorch_to_caffe.trans_net(model.model, input_variable, self.name)
        self.caffe_prototxt = "{}.prototxt".format(self.name)
        self.caffe_model =  "{}.caffemodel".format(self.name)
        self.engine_model = "{}.engine".format(self.name)
        pytorch_to_caffe.save_prototxt('{}/{}'.format(model_transform_result_dir,self.caffe_prototxt))
        pytorch_to_caffe.save_caffemodel('{}/{}'.format(model_transform_result_dir,self.caffe_model))

        caffe_op_dir = "./pytorch2caffe"
        sys.path.append(caffe_op_dir)
        trans_module = 'caffe_prototxt_op'
        caffe_op = __import__(trans_module)

        self.output_blobs, self.output_names = caffe_op.getOutputBlobsandNames('{}/{}'.format(model_transform_result_dir,self.caffe_prototxt))
        #print(self.output_blobs)
        #print(self.output_names)
        #generate caffe2tensorRT param
        self.saveCaffe2tensorRTParam()


    def transform2tensorRT(self):
        ExecutablePath = '../build/translate/translate '
        output_dir = self.model_transform_result_dir
        cmd  = ExecutablePath+output_dir
        os.system(cmd)


def main(argv):
    # config_path = "./test.conf"
    # pre_process = PreProcess(config_path)
    # image = cv2.imread("1.bmp")
    # img=pre_process.process_image(image)
    # print(img.shape)
    # print(img)
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "method",
    #     help="method"
    # )
    parser.add_argument(
        "model_class_dir",
        help="model_class_dir"
    )
    args = parser.parse_args()
    model_class_dir = args.model_class_dir
    transform = Transform(model_class_dir)
    transform.transform2caffe()
    transform.transform2tensorRT()
    # output = convertPytorch2caffemodel(config_path)
    # ExecutablePath = './sampleTrack'


if __name__ == "__main__":
    import sys
    main(sys.argv)
