#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 18:49:13 2018

@author: tymarking

>>> a = np.matrix('1 2; 3 4')
>>> print(a)
[[1 2]
 [3 4]]

"""

"""
dimensions as (inputs, L1, L2... Ln, outputs)

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
            rands.append(random.gauss(0, 0.5))
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

"""
precondition: inputs is a list of the first activations

"""
def run(inputs, net):
    #1: multiply weights by prev activations
    #2: apply non-linear (sigmoid or Relu)
    #3: repeat 1-2 for all layers
    #4: return last activation
    a = np.reshape(inputs, (len(inputs),1))
    a = np.matrix(a)
    for layer in net:
        a = sigmoid(layer[0]*a+layer[1])
    return a
    