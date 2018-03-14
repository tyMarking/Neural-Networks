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

"""
precondition: inputs is a list of the first activations, net is the net

"""
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
precondition: trainSet is tuple of (inputs, correct)
postcondition: return tuple (weights, biases) of the positive gradiant
"""
def train(net, trainSet):
    
    #1: multiply weights by prev activations
    #2: apply non-linear (sigmoid or Relu)
    #3: repeat 1-2 for all layers
    #4: return last activation
    
    #final grads list (not averaged)
    grads = []
    
    #for manual eval
    right = 0
    wrong = 0
    
    for ins in trainSet:
        #sets inputs in vertical matrix
        a = np.reshape(ins[0], (len(ins[0]),1))
        a = np.matrix(a)
        
        #forward propigation
        currentVals = a
        vals = []
        for layer in net:
            vals.append(currentVals)
            currentVals = sigmoid(layer[0]*currentVals+layer[1])
            
        
        finalVals = []
        for val in currentVals.tolist():
            finalVals.append(val[0])
        
        largest = finalVals[0]
        largestIndex = 0
        for i in range(len(finalVals)):
            if finalVals[i] > largest:
                largest = finalVals[i]
                largestIndex = i
        if largestIndex == ins[1]:
            right += 1
        else:
            wrong += 1
        
        
        #backwards prop
        #cost function for initial derivatives
        #C =  * sum(y1-y2)^2
        #dc/dy = -2(y-y)
        initGrads = []
        for i in range(len(finalVals)):
            if i == ins[1]:
                y = 1
            else:
                y = 0
            initGrads.append(-2*(y-finalVals[i]))
        listVals = []
        for val in vals:
            listVals.append(val.tolist()[0])
            
        #turning net into lists, ussage: weights[L][i][j]
        weights = []
        biases = []
        for layer in net:
            weights.append(layer[0].tolist())
            biases.append(layer[1].getT().tolist()[0])
        
        """
        newB = []
        for bias in biases:
            bLayer = []
            for layer in bias:
                bLayer.append(layer[0])
            newB.append(bLayer)
        biases = newB
        
        newW = []
        for weight in weights:
            wLayer = []
            for layer in weight:
                wLayer.append(layer[0])
            newW.append(wLayer)
        weights = newW
        """
        
        #node in layer L = i
        #node in layer (L-1) = j

        #grad structure like net but no matrix. 
        #list[(weights,biases), (w,b), (w,b)] : Length = # of layers
        #weights = list[L][i][j]
        #biases = list[L][i]
        #activations = listVals[L][i]
        
        #final grads list (not averaged)
        #grads = [([],[])] * len(net)
    
        #just to get the right shape/size
        
        wGrads = weights.copy()
        bGrads = biases.copy()
#        aGrads.append(initGrads)
        prevAGrads = initGrads
        
        #for each a Grad in layer L+1
        #for each node in layer L
        #da = da(L+1) * dSigmoid
        #for each weight dwi = da*ai(L-1)
        #for each a(L-1) dai = da*wi
        #for each bias db = da
        
        #L = layer number, from out to 0 in (in,0,1,2,out)
        for L in reversed(range(len(net))):
            
            aGrads = []
            #node in L
            for i in range(len(listVals[L])):
                #h = node above
                subAGrads = []
                for h in range(len(prevAGrads)):
                    a = listVals[L][i]
                    #this a grad = prevAGrad * connecting weight * sigmoid derivative
                    subAGrads.append(prevAGrads[h] * weights[L][h][i] * ( (1-a)*a ))
                    # weight grads = prevAGrad * this a
                    wGrads[L][h][i] = prevAGrads[h] * a
                    
                aGrad = sum(subAGrads)/len(subAGrads)
                aGrads.append(aGrad)
                bGrad = aGrad
                bGrads[L][i] = bGrad
            
            prevAGrads = aGrads
            
        """  
        newBGrads = []
        for bGrad in bGrads:
            bLayer = []
            for layer in bGrad:
                bLayer.append(layer[0])
            newBGrads.append(bLayer)
        bGrads = newBGrads
        """
        
        grads.append((wGrads, bGrads))
        
#        layers = []
#        for i in range(len(wGrads)):
#            layers.append((wGrads[i],bGrads[i]))
#        grads.append(layers)
        

    #average the gradiants

#    #empty matrix of right sizes
#    #print(grads[0][0])
##    print("SPACE")
#    wSum = np.matrix(grads[0][0])
#    bSum = np.matrix(grads[0][1])
##    wSum = wSum - wSum
##    bSum = bSum - bSum
##    print(wSum)
##    print(grads[0])
#    for grad in grads[1:]:
#        wSum = wSum + np.matrix(grad[0])
#        bSum = bSum + np.matrix(grad[1])
    
#    print("test")
#    wAvg = wSum / len(grads)
#    bAvg = bSum / len(grads)
    wAvg = grads[0][0]
    bAvg = grads[0][1]
    
    for L in range(len(grads[0][0])):
        for i in range(len(grads[0][0][L])):
            for j in range(len(grads[0][0][L][i])):
                wSum = 0
                for g in range(len(grads)):
                    wSum += grads[g][0][L][i][j]
                
                wAvg[L][i][j] = wSum / len(grads)
    for L in range(len(grads[0][1])):
        for i in range(len(grads[0][1][L])):
            bSum = 0
            for g in range(len(grads)):
                bSum += grads[g][1][L][i]
                
            bAvg[L][i] = bSum / len(grads)

    
#    for L in range(len(wList)):
#        for i in range(len(wList[L])):
#            for j in range(len(wList[i])):
#                wList[L][i][j] /= len(wList)
#                
#    for L in range(len(bList)):
#        for i in range(len(bList[L])):
#            bList[L][i] /= len(bList)
#        
#    print("SPACE!")
#    wAvg = np.matrix(wAvg)
#    bAvg = np.matrix(bAvg)
    
    
    finalGrad = []
    for i in range(len(wAvg)):
        finalGrad.append((wAvg[i],bAvg[i]))
    
    return ((finalGrad, (right/(right+wrong))))
