#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:05:07 2018

@author: 136029
"""

import numpy as np
import random

import math

#helper function
def sigmoid(x):
    xList = x.tolist()
    newList = []
    for subList in xList:
        newSubList = []
        for item in subList:
            newSubList.append(1 / (1 + math.exp(-item)))
        newList.append(newSubList)
    return np.matrix(newList)

def dsigList(x):
    ret = []
    for z in x:
        ret.append(1-z)*z
    return ret

def dSig(x):
    xList = x.tolist()
    newList = []
    for subList in xList:
        newSubList = []
        for item in subList:
            sigX = (1/(1+math.exp(-item)))
            newSubList.append(sigX * (1 - sigX))
        newList.append(newSubList)
    return np.matrix(newList)

""" New Net
precondition: dimensions is an array of form (inputs, L1, L2... Ln, outputs). Length must be at least 3
postcondition: returns list of tuples of type (weights, biases). 
This list is one shorter than dimensions as the input layer does not need weights or biases
"""
def newNet(dimensions):
    print("Generating a new neural net")
    if len(dimensions) < 3:
        print("ERROR: newNet dimensions must be at least 3")
        return 
    layers = []
    for i in range(1,len(dimensions)):
        dim = dimensions[i]
        prevNodes = dimensions[i-1]
        
        #forming weight matrix
        rands = []
        for j in range(prevNodes*dim):
            rands.append(random.gauss(0, 2))
        rands = np.reshape(rands, (dim,prevNodes))
        weights = np.matrix(rands)
        
        #forming biase matrix
        rands = []
        for j in range(dim):
            rands.append(random.gauss(0, 0.5))
        rands = np.reshape(rands, (dim,1))
        biases = np.matrix(rands)
        
        #adding (weights, biases) to layers
        layers.append((weights,biases))
    print("New neural net generated")
    return layers


def run(net, inputs):

    
    a = np.reshape(inputs, (len(inputs),1))
    a = np.matrix(a)

    for i in range(len(net)):
        a = sigmoid(net[i][0]*a+net[i][1])
#    a = net[len(net)][0]*a+net[len(net)][1]
    a = a.tolist()
    currentMax = a[0]
    index = 0
    for i in range(len(a)):
        if a[i] > currentMax:
            currentMax = a[i]
            index = i
    return index


"""
    trainset = [(image, label),(img,lbl)...]
    
"""
def train(net, trainSet):
    gs = []
    right = 0
    wrong = 0

    for trainer in trainSet:
        image = trainer[0]
        label = trainer[1]
        
        """ FORWARD PROPIGATION """
        activations = np.matrix(image).getT()
        print(activations)

        aStore = []
        zStore = []
        aStore.append(activations)
        for layer in net:
            oneZ = layer[0] * activations + layer[1]
            activations = sigmoid(oneZ)
            aStore.append(activations)
            zStore.append(oneZ)
            
        aList = activations.getT().tolist()
        aList = aList[0]
        maxIndex = 0
        for i in range(len(aList)):
            if aList[i] > aList[maxIndex]:
                maxIndex = i
        
        
        if (label == aList[maxIndex]):
            right += 1
        else:
            wrong += 1
        
        """ BACKPROPIGATION """
        
        """
        #Initial partial gradiants of the output neurons
        initGs = []
        for i in range(len(activations)):
            if i == label:
                y = 1
            else:
                y = 0
            g = (-1*(y-aList[i])) / (1 + math.exp(-aList[i]))

            initGs.append(g)
        print(aList)
        print(initGs)
        """
        
        """
        for node i in layer L-1
        for node j in layer L
        compute inter weight grad
        compute part-activation grad
        average part-activations
        
        compute bias somewhere?
        
        """
        
#        for L in reversed(range(len(aStore))):
        
        ys = [0] * len(aList)
        ys[label] = 1
        
        y = np.matrix(ys).getT()
        
        error = np.multiply((aStore[-1]-y), dSig(zStore[-1]))
        print("JIIIII")
        print(error)
            
        
        



