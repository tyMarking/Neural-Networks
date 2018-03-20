# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 14:16:22 2018

@author: Ty
"""
import MyNeuralNet as NN
#stuff

import SecondNeuralNet as NN2

import gzip
import json
import numpy as np
import pylab




net = NN2.newNet((3,2,10))
NN2.train(net, [([1,2,3],5)])


#read the MNIST data
print("Reading MNIST data")
trainImages = gzip.open("data/train-images-idx3-ubyte.gz", 'rb')
trainLabels = gzip.open("data/train-labels-idx1-ubyte.gz", 'rb')
imagesBytes = []


trainImages.read(16)
trainLabels.read(8)
trainData = []

#should be 60000
for i in range(60000):
    image = []
    for pixle in trainImages.read(728):
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



def trainAndUpdate(file, trainSet):
    #learning coeficient
    lC = 0.1
    net = loadFromFile(file)
    ret = NN.train(net,trainSet)
    grad = ret[0]
    percent = ret[1]
    #print(percent)
    netList = []
    for layer in net:
        netList.append((layer[0].tolist(),layer[1].tolist()))
#    print(netList)
        
    for L in range(len(netList)):
        
        #weights
        for i in range(len(netList[L][0])):
            for j in range(len(netList[L][0][i])):
                netList[L][0][i][j] += (-lC) * grad[L][0][i][j]
        
        #biases
        for i in range(len(netList[L][1])):
            netList[L][1][i][0] += (-lC) * grad[L][1][i]
    
    #converting back to matrix form
    matrixList = []
    for layer in netList:
        matrixList.append((np.matrix(layer[0]),np.matrix(layer[1])))
    
    net = matrixList
    saveToFile(net, file)
    
    return percent

#create the inital net
dimensions = (728,16,16,10)
net = NN.newNet(dimensions)
#net = loadFromFile("firstNetSmall2.txt")
saveToFile(net, "firstNetSmall4.txt")







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


