# Brian Blythe and Parker Bruni
# CS 434
# Implementation Assignment 2
# Problem 2

import numpy as np
import random
import math

#format output to be more readable
np.set_printoptions(suppress=True)

class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None

    self.parent = None
    self.depth = None
    self.currentlist = []
    
#------------------------------------------
# Function: loadData
# Description: loads the csv files
#------------------------------------------

def loadData():

    #load all the training data into an array X
    Xdata = np.genfromtxt ('knn_train.csv', delimiter=",")
    X = np.matrix(Xdata[:, 1:])

    #load all the testing data into an array Xtest
    Xtestdata = np.genfromtxt ('knn_test.csv', delimiter=",")
    Xtest = np.matrix(Xtestdata[:, 1:])

    #set Y array to the last column of X data, transpose Y 
    Y = np.matrix(Xdata[:,0]).T
    Ytest = np.matrix(Xtestdata[:,0]).T

    return X, Xtest, Y, Ytest

#------------------------------------------
# Function: entropy
# Description: Calculates the entropy of
#              of a split based on the
#              number of 1s and -1s input
#              as being on each side of it
#------------------------------------------

def entropy(leftOfSplit, rightOfSplit):

    total = leftOfSplit + rightOfSplit

    if leftOfSplit != 0 and rightOfSplit != 0:
        entr = -float(leftOfSplit)/total * math.log(float(leftOfSplit)/total,2) \
               -float(rightOfSplit)/total * math.log(float(rightOfSplit)/total,2)
    elif leftOfSplit == 0 and rightOfSplit != 0:
        entr = -float(rightOfSplit)/total * math.log(float(rightOfSplit)/total,2)
    elif leftOfSplit != 0 and rightOfSplit == 0:
        entr = -float(leftOfSplit)/total * math.log(float(leftOfSplit)/total,2)
    else:
        entr = 0

    return entr

#------------------------------------------
# Function: calculateSplit
# Description: Calculates the location of the maximum gain,
#              what that gain is, and the number of 1s and
#              -1s on each side
#------------------------------------------

def calculateSplit(column, Y):

    newarray = np.concatenate((Y, column), axis = 1) #concatenates the column with the training Y values
    sortedarray = newarray[np.lexsort(np.fliplr(newarray).T[0])][0,:,:] # sort array by the column parameter

    numrows = X.shape[0]

    gainset = []
    
    for i in range(numrows):

        splitL, splitR = sortedarray[:i,:], sortedarray[i:,:] #binary split array by i

        posL = np.count_nonzero(splitL == 1)
        negL = np.count_nonzero(splitL == -1)
        posR = np.count_nonzero(splitR == 1)
        negR = np.count_nonzero(splitR == -1)

        entropyM = entropy(posL + posR, negL + negR) #entropy of parent
        entropyL = entropy(posL, negL) #entropy of left child
        entropyR = entropy(posR, negR) #entropy of right child
        gain = infoGain(i, numrows, entropyM, entropyL, entropyR) #gain of split
        gainset = gainset + [gain]

    idx_max_splt = gainset.index(max(gainset)) #split index of max gain
    maxgain = gainset[idx_max_splt] #the gain at this index
    
    return posL, negL, posR, negR, idx_max_splt, maxgain


#------------------------------------------
# Function: infoGain
# Description: calculates the information gain
#------------------------------------------

def infoGain(index, numrows, entropyM, entropyL, entropyR):

    gain = entropyM - (float(index)/numrows*entropyL) - ((1-float(index)/numrows)*entropyR)

    return gain

    
#------------------------------------------
# Function: decistionTree
# Description: Runs the decision Tree algorithm for a single stump
#------------------------------------------

def decisionTree(X, Xtest, Y, Ytest, d):
    '''
    cycle through all versions with the top node of the decision tree
    being a new column of data
    '''

    if d == 0:
        return X

    totals = []

    numcols = X.shape[1]

    for j in range(d):
        print "Considering depth: ", j+1

        for i in range(numcols): 
            posL, negL, posR, negR, idx_max_splt, maxgain = calculateSplit(X[:,i], Y) 
            totals = totals + [[i, idx_max_splt, maxgain]] #add values to list

        arr = np.array(totals) #make into numpy array

        maxindex, maxsplt, maxgain = arr.max(axis=0) 
        location = arr.argmax(axis=0)[2]
        print "Feature number: ", location
        print "Split at index: ", arr[location,1]
        print "Gain: ", maxgain

        XplusY = 

        sortedX = X[np.lexsort(np.fliplr(X).T[numcols-location])][0,:,:] #sort by chosen parameter
        sortedXL, sortedXR = sortedX[:int(arr[location,1]),:], sortedX[int(arr[location,1]):,:] #binary split by index

        FinalSplitL = decisionTree(sortedXL, Xtest, Y, Ytest, d-1)
        
        FinalSplitR = decisionTree(sortedXR, Xtest, Y, Ytest, d-1)

        #print sortedXL, "\n", sortedXR


#-------------------------------------------
# Function: MAIN
#-------------------------------------------

X, Xtest, Y, Ytest = loadData()

#Normalize columns of numpy arrays

X_normed = X / X.max(axis=0)
Xtest_normed = Xtest / X.max(axis=0)

d = 2

decisionTree(X_normed, Xtest_normed, Y, Ytest, d)

