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
    neighbors = dist.argsort()[:k] #returns the indices of the k nearest neighbors
	
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
# Function: kNeighborTrain
# Description: Runs K nearest neighbor
#              algorithm on training data
#-------------------------------------------

def kNeighborTrain(k, X, Xtest, Y, Ytest):

    numtestrows = Xtest.shape[0]
    guesses = []
    
    #for each test row, find the k nearest neighbors in the learned data set (X)
    for i in range(numtestrows):
        neighbors = getNeighbors(X[i,:], X, k) #get the k nearest neighbors
        vote = neighborVote(neighbors, Y, k) #get the vote of the k neighbors
        guesses = guesses + [vote] #inefficient in numpy but I don't care
        
    return guesses

#-------------------------------------------
# Function: kNeighborTest
# Description: Runs K nearest neighbor
#              algorithm on test data
#-------------------------------------------

def kNeighborTest(k, X, Xtest, Y, Ytest):

    numtestrows = Xtest.shape[0]
    guesses = []
    
    #for each test row, find the k nearest neighbors in the learned data set (X)
    for i in range(numtestrows):
        neighbors = getNeighbors(Xtest[i,:], X, k) #get the k nearest neighbors
        vote = neighborVote(neighbors, Y, k) #get the vote of the k neighbors
        guesses = guesses + [vote] #inefficient in numpy but I don't care
        
    return guesses 

 
#-------------------------------------------
# Function: trainError
# Description: Returns percentage of correct
#				guesses of the training dataset
#-------------------------------------------
def Error(Y, guesses):
	
	if len(Y) != len(guesses):
		print "ERROR: length of data values array different size than guesses array"
	
	numCorrect = 0
	#compare each actual value in Y to its corresponding guessed value
	
	
	for i in range (0, len(guesses)):
		if Y.item(i) == float(guesses[i]):
			numCorrect = numCorrect + 1
	
	#numberErrors = len(guesses) - numCorrect
	#return numberErrors
	
	#compute and return the percentage of correct guesses for this trial
	return float(numCorrect) / len(guesses)
	
	
	
#-------------------------------------------
# Function: MAIN
#-------------------------------------------

X, Xtest, Y, Ytest = loadData()

#Normalize columns of numpy arrays

X_normed = X / X.max(axis=0)
Xtest_normed = Xtest / X.max(axis=0)

#K nearest neighbor algorithm


'''
Each row in X_normed represents a 30 dimensional point.
We need to compare every row point in Xtest_normed to
each k closest row points of X_normed
'''
ftrain = open('trainOutputPCorrect.csv', 'w+')
ftest = open('testOutputPCorrect.csv', 'w+')
fcross = open('crossValidationError.csv', 'w+')

#try k values ranging from 1 to 51
for k in range(1, 285):

	print k
	k_result_Train = kNeighborTrain(k, X_normed, Xtest_normed, Y, Ytest)
	k_result_Test = kNeighborTest(k, X_normed, Xtest_normed, Y, Ytest)	
	
	TrainPercentageCorrect = Error(Y, k_result_Train)
	TestPercentageCorrect = Error(Ytest, k_result_Test)
	
	ftrain.write(str(TrainPercentageCorrect) + "\n")
	ftest.write(str(TestPercentageCorrect) + "\n")
	
	print "Train Correct: ", TrainPercentageCorrect
	print "Test Correct: ", TestPercentageCorrect


ftrain.close()
ftest.close()
fcross.close()



