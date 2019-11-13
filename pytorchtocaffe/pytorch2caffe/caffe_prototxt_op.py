#!/usr/bin/python
#_*_ coding:UTF-8 _*_
#pytorch version = 1.0.0
from __future__ import print_function
import torch
import torch.nn as nn
from Caffe import caffe_net
import torch.nn.functional as F
from torch.autograd import Variable
from Caffe import layer_param
from torch.nn.modules.utils import _pair
import numpy as np
'''
def getOutputBlobs(prototxt_file):
	prototxt=caffe_net.Prototxt(prototxt_file)
	bottoms={}
	tops={}
	#print(prototxt.layers())
	for layer in prototxt.layers():
		# help(layer)
		for bottom in layer.bottom:
			bottoms[bottom]=1
		for top in layer.top:
			tops[top]=1
		#print(layer.name)
	#print(bottoms)
	#print(tops)
	only_top=[]
	for key in tops:
		if (key not in bottoms):
			only_top.append(key)
	print(only_top)
	name_list = []
	for outname in prototxt.layers():
		#print(outname.top[0])
		for top in only_top:
			#print(top)
			if (top == outname.top[0]):
				#print(outname.name)
				name_list.append(outname.name)
	print(name_list)

	# print(only_top)
	return only_top

def main(argv):
	# getOutputBlobs()
	getOutputBlobs("")


if __name__ == "__main__":
    import sys
    main(sys.argv)
'''
'''
def getOutputBlobsandNames(prototxt_file):
	bottoms = {}
	tops = {}
	only_top = []
	name_list = []
	prototxt = caffe_net.Prototxt(prototxt_file)
	for layer in prototxt.layers():
		for bottom in layer.bottom:
			bottoms[bottom] = 1
		for top in layer.top:
			tops[top] = 1
	for key in tops:
		if (key not in bottoms):
			only_top.append(key)
	print(bottoms)
	print(tops)
	print(only_top)
	for outname in prototxt.layers():
		for top in only_top:
			if (top == outname.top[0]):
				name_list.append(outname.name)
	return only_top, name_list
'''

def getOutputBlobsandNames(prototxt_file):
	only_top = []
	output_name = []
	prototxt = caffe_net.Prototxt(prototxt_file)
	#print(type(prototxt.layers()))
	#print(prototxt.layers()[0].top[0])
	number = len(prototxt.layers())
	#print(len(prototxt.layers()))
	temp_top = 0
	temp_bottom = 1
	while(temp_top < number - 1):
		while(temp_bottom < number):

			if (prototxt.layers()[temp_top].top[0] in prototxt.layers()[temp_bottom].bottom):
				break
			if (temp_bottom == number - 1):
				only_top.append(prototxt.layers()[temp_top].top[0])
				output_name.append(prototxt.layers()[temp_top].name)
			temp_bottom += 1

		temp_top += 1
		temp_bottom = temp_top + 1
	only_top.append(prototxt.layers()[number - 1].top[0])
	output_name.append(prototxt.layers()[number - 1].name)
	#print(only_top)
	#print(output_name)
	return only_top, output_name

if __name__ == "__main__":
	path = "/data/wuh/project/model_transform_test/trans_result/vehicle_model_new/model_model.prototxt"
	getOutputBlobsandNames(path)
