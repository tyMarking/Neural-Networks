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

def dsigList(x):
    ret = []
    for z in x:
        ret.append(1-z)*z
    return ret
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
def run(inputs, net, correct):
    #1: multiply weights by prev activations
    #2: apply non-linear (sigmoid or Relu)
    #3: repeat 1-2 for all layers
    #4: return last activation
    a = np.reshape(inputs, (len(inputs),1))
    a = np.matrix(a)

    for i in range(net-1):
        a = sigmoid(net[i][0]*a+net[i][1])
    a = net[len(net)][0]*a+net[len(net)][1]
    return a

def train(inputs, net, rights):
    
    #1: multiply weights by prev activations
    #2: apply non-linear (sigmoid or Relu)
    #3: repeat 1-2 for all layers
    #4: return last activation
    
    #final grads list (not averaged)
    grads = [([],[])] * len(net)
        
    for input in inputs:
        #sets inputs in vertical matrix
        a = np.reshape(inputs, (len(inputs),1))
        a = np.matrix(a)
        
        #forward propigation
        currentVals = a
        vals = []
        for layer in net:
            vals.append(currentVals)
            currentVals = sigmoid(layer[0]*currentVals+layer[1])
            
        
        finalVals = currentVals
        #cost function for initial derivatives
        #C =  * sum(y1-y2)^2
        #dc/dy = -2(y-y)
        initGrads = []
        for i in range(finalVals):
            if i == right:
                y = 1
            else:
                y = 0
            initGrads.append(-2*(y-finalVals[i]))
        listVals = []
        for val in vals:
            listVals.append(val.tolist)
            
        #turning net into lists, ussage: weights[L][i][j]
        weights = []
        biases = []
        for layer in net:
            weights.append(layer[0].getT().tolist())
            biases.append(layer[1].getT().tolist())
            
            
        #node in layer L = i
        #node in layer (L-1) = j

        #grad structure like net but no matrix. 
        #list[(weights,biases), (w,b), (w,b)] : Length = # of layers
        #weights = list[i][j]
        #biases = list[i]
        #activations = listVals[L][i]
        
        #final grads list (not averaged)
        #grads = [([],[])] * len(net)
    
        
        wGrads = []
        bGrads = []
#        aGrads.append(initGrads)
        prevAGrads = initGrads
        
        #for each a Grad in layer L+1
        #for each node in layer L
        #da = da(L+1) * dSigmoid
        #for each weight dwi = da*ai(L-1)
        #for each a(L-1) dai = da*wi
        #for each bias db = da
        
        #L = layer number, from out to 0 in (in,0,1,2,out)
        for L in reversed(range(1,len(listVals))):
            
            aGrads = []
            #node in L
            for i in range(len(listVals[L])):
                #h = node above
                subAGrads = []
                for h in range(len(prevAGrads)):
                    a = listVals[L][i]
                    #this a grad = prevAGrad * connecting weight * sigmoid derivative
                    subAGrads.append(prevAGrads[h] * weights[h][i] * ( (1-a)*a ))
                    # weight grads = prevAGrad * this a
                    wGradHI = prevAGrads[h] * a
                aGrad = sum(subAGrads)/len(subAGrads)
                aGrads.append(aGrad)
            
            
            
            
            
            
            
            
            nextAGrads = [[]]*len(listVals[L])
            for prevAGrad in prevAGrads:
                
                #nodes in layer L
                for i in range(len(listVals[L])):
                    a = listVals[L][i]
                    aGrad = prevAGrad * ( (1-a)*a )
                    nextAGrads[i].append(aGrad)

                    #nodes in layer (L-1)
                    for j in range(len(listVals[L-1])):
                        
            #avergae nextAGrads
            aGrads = []
            for aGrad in nextAGrads:
                aGrads.append(sum(aGrads)/len(aGrads))
            
                
            
            
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    """
    for l in reversed(range(len(net))):
        layer = net[l]
        weights = layer[0]
        bias = layer[1]
        val = vals[l]
        ddot = dSigList(val)
        #nodes
        for i in range(ddot):
            da = []
            dw = []
            #connections
            for j in range(weights):
                da.append(ddot[i]*weights[i][j])
                dw.append(ddot[i]*prev)
     """
          
    #backwards prop
    return a
