# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 14:16:22 2018

@author: Ty
"""
import MyNeuralNet as NN
import gzip
import json
import numpy as np


#Helper functions
def saveToFile(net, file):
    netList = []
    for layer in net:
        netList.append((layer[0].tolist(),layer[1].tolist()))
    netJson = json.dumps(netList)
    file = open(file, "w")
    file.truncate(0)
    file.write(netJson)

def loadFromFile(file):
    file = open(file, "r")
    netJson = file.read()
    netList = json.loads(netJson)
    matrixList = []
    for layer in netList:
        matrixList.append((np.matrix(layer[0]),np.matrix(layer[1])))
    return matrixList
#create the inital net
dimensions = (728,16,16,10)
#net = NN.newNet(dimensions)
net = loadFromFile("firstNet.txt")
#read the MNIST data
print("Reading MNIST data")
trainImages = gzip.open("data/train-images-idx3-ubyte.gz", 'rb')
trainLabels = gzip.open("data/train-labels-idx1-ubyte.gz", 'rb')
imagesBytes = []


trainImages.read(16)
trainLabels.read(8)
trainData = []

for i in range(60000):
    image = []
    for pixle in trainImages.read(728):
        image.append(pixle/255)
    label = trainLabels.read(1)[0]
    trainData.append((image, label))

print("Finished reading MNIST data")

print(NN.run(trainData[0][0], net))
print(trainData[0][1])



saveToFile(net, "firstNet.txt")