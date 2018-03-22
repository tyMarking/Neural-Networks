# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 10:39:15 2018

@author: 136029
"""


import SecondNeuralNet as NN2

import gzip
import json
import numpy as np
import pylab


#read the MNIST data
print("Reading MNIST data")
trainImages = gzip.open("data/train-images-idx3-ubyte.gz", 'rb')
trainLabels = gzip.open("data/train-labels-idx1-ubyte.gz", 'rb')
imagesBytes = []


trainImages.read(16)
trainLabels.read(8)
trainData = []

#should be 60000
for i in range(100):
    image = []
    for pixle in trainImages.read(784):
        image.append(pixle/255)
    label = trainLabels.read(1)[0]
    trainData.append((image, label))

print("Finished reading MNIST data")

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




file = "2ndNNv.1.json"

net = NN2.newNet((784,16,16,10))
#grad, percent = NN2.train(net, [([1,2,3],2), ([2,3,4],1)])
#print(percent)
saveToFile(net,file)



percents = []
currentIndex = 0
while True:
    if currentIndex == 100:
           pylab.figure("1")
           pylab.clf()
           pylab.title("Perfromance")
           pylab.xlabel("Per train")
           pylab.ylabel("Percent")
           #pylab.ylim(0,maxPop)
           pylab.plot(range(len(percents)),percents)
           pylab.show()
           #print("Average Percent: " + str(sum(percents)/len(percents)))
           print("Current Percent: " +  str(percents[-1]))
    trainSet = []
    
    for i in range(100):
        currentIndex = currentIndex % 100
        trainSet.append((trainData[i+currentIndex]))
    
    currentIndex += i + 1
    
    net, percent = NN2.train(net, trainSet)
    percents.append(percent)
    
    saveToFile(net, file)



"""

#print(NN.run(net, trainData[0][0]))
#print(trainData[0][1])
percents = []
currentIndex = 0
while True:
    if currentIndex == 100:
           pylab.figure("1")
           pylab.clf()
           pylab.title("Perfromance")
           pylab.xlabel("Per train")
           pylab.ylabel("Percent")
           #pylab.ylim(0,maxPop)
           pylab.plot(range(len(percents)),percents)
           pylab.show()
           print("Average Percent: " + str(sum(percents)/len(percents)))
    trainSet = []
    
    for i in range(100):
        currentIndex = currentIndex % 100
        trainSet.append((trainData[i+currentIndex]))
    percents.append(trainAndUpdate("firstNetSmall4.txt", trainSet))
    currentIndex += i + 1
    
    
    #saveToFile(net, "firstNet.txt")

"""
