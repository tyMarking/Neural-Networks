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
import random as rnd


#read the MNIST data
print("Reading MNIST data")
#training data
trainImages = gzip.open("data/train-images-idx3-ubyte.gz", 'rb')
trainLabels = gzip.open("data/train-labels-idx1-ubyte.gz", 'rb')
imagesBytes = []


trainImages.read(16)
trainLabels.read(8)
trainData = []

#should be 60000
for i in range(60000):
    image = []
    for pixle in trainImages.read(784):
        image.append(pixle/255)
    label = trainLabels.read(1)[0]
    trainData.append((image, label))


testImages = gzip.open("data/t10k-images-idx3-ubyte.gz", 'rb')
testLabels = gzip.open("data/t10k-labels-idx1-ubyte.gz", 'rb')
imagesBytes = []

#testing data
testImages.read(16)
testLabels.read(8)
testData = []

#should be 10000
for i in range(10000):
    image = []
    for pixle in testImages.read(784):
        image.append(pixle/255)
    label = testLabels.read(1)[0]
    testData.append((image, label))
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




file = "NN2v.64.json"

net = NN2.newNet((784,64,64,10))
saveToFile(net, file)

#net = loadFromFile(file)


def train(file):
    net = loadFromFile(file)
    
    
    percent = 0
    currentIndex = 0
    while True:
        
        trainSet = []
        for i in range(100):
            currentIndex = currentIndex % 60000
#            trainSet.append((trainData[rnd.randint(0,60000-1)]))
            trainSet.append((trainData[i+currentIndex]))
        
        currentIndex += i + 1
        if currentIndex % 1000 == 0:
               """
               pylab.figure("1")
               pylab.clf()
               pylab.title("Perfromance")
               pylab.xlabel("Per train")
               pylab.ylabel("Percent")
               pylab.plot(range(len(percents)),percents)
               pylab.show()
               """
               #print("Average Percent: " + str(sum(percents)/len(percents)))
               print("Current Percent: " +  str(percent))
        
        net, percent = NN2.train(net, trainSet, 2)
        
        saveToFile(net, file)

def test(file):
    net = loadFromFile(file)
    right = 0
    wrong = 0
    for i in range(len(testData)):
        if NN2.run(net, testData[i][0]) == testData[i][1]:
            right += 1
        else:
            wrong += 1
    
    print(str(100*right/(right+wrong)) + " percent correct")

#
#test(file)
train(file)
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
