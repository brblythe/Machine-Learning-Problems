# Brian Blythe and Parker Bruni
# CS 434
# Implementation Assignment 2
#


import numpy as np
import random
import math

#format output to be more readable
np.set_printoptions(suppress=True)

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

#-------------------------------------------
# Function: euclideanDistance
# Description: Finds euclidean distances
#              of all elements of X from
#              a single test row
#-------------------------------------------

def euclideanDistance(testrow, X):
    dist = np.linalg.norm(X-testrow, axis=1) #computes squareroot of squared differences (i.e. the norm)
    return dist

#-------------------------------------------
# Function: getNeighbors
# Description: gets the K nearest neighbors
#-------------------------------------------

def getNeighbors(testrow, X, k):

    dist = euclideanDistance(testrow, X) #returns euclidean distance of the test row from each row of X
    neighbors = dist.argsort()[-k:][::-1] #returns the indices of the k nearest neighbors

    return neighbors        
#-------------------------------------------
# Function: neighborVote
# Description: returns the vote of the k
#              nearest neighbors
#-------------------------------------------

def neighborVote(neighbors, Y, k):

    numnegative = np.histogram(Y[neighbors])[0][0] #number of -1 votes
    numpositive = k - numnegative #number of 1 votes

    if numnegative > numpositive:
        vote = -1
    elif numnegative < numpositive:
        vote = 1
    else:
        vote = int(math.ceil(random.uniform(-1, 1))) #if its a draw, randomly assign -1 or 1
        if vote == 0:
            vote = -1
    
    return vote

#-------------------------------------------
# Function: kNeighbor
# Description: Runs K nearest neighbor
#              algorithm
#-------------------------------------------

def kNeighbor(k, X, Xtest, Y, Ytest):

    numtestrows = Xtest.shape[0]
    guesses = []
    
    #for each test row, find the k nearest neighbors in the learned data set (X)
    for i in range(numtestrows):
        neighbors = getNeighbors(Xtest[i,:], X, k) #get the k nearest neighbors
        vote = neighborVote(neighbors, Y, k) #get the vote of the k neighbors
        guesses = guesses + [vote] #innefficient in numpy but I don't care
        
    return guesses

#-------------------------------------------
# Function: costFunction
# Description: Calculates costs to evaluate
#              best location for splits
#-------------------------------------------

def costFunction(X, Xtest, Y, Ytest):

    pass
    

#-------------------------------------------
# Function: decisionTree
# Description: Runs descision tree algorithm
#-------------------------------------------

def decisionTree(d, X, Xtest, Y, Ytest):

    
    
    '''
    - Choose the best test to be the root of the tree.
    - Create a descendant node for each test outcome
    - Examples in training set S are sent to the appropriate
    descendent node based on the test outcome
    - Recursively apply at each descendant node to select the
    best attribute to test using its subsest of training samples
    - If all samples in a node belong to the same class, turn it
    into a leaf node of that class
    '''
    pass

    
        
#-------------------------------------------
# Function: MAIN
#-------------------------------------------

X, Xtest, Y, Ytest = loadData()

#Normalize columns of numpy arrays

X_normed = X / X.max(axis=0)
Xtest_normed = Xtest / Xtest.max(axis=0)

#K nearest neighbor algorithm

k = 11 # set this value for k

'''
Each row in X_normed represents a 30 dimensional point.
We need to compare every row point in Xtest_normed to
each row of X_normed
'''

k_result = kNeighbor(k, X_normed, Xtest_normed, Y, Ytest)

print k_result



