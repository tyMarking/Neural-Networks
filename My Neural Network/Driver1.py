# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 14:16:22 2018

@author: Ty
"""
import MyNeuralNet as NN

dimensions = (784,16,16)
print(NN.newNet(dimensions))
file1 = open("data2/train-labels-idx1-ubyte.gz", 'rb')
imagesBytes = []
print(file1.read(36))
for b in file1.read():
    imagesBytes.append(b)
print(imagesBytes[0:64])