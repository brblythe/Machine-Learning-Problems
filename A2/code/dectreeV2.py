# Brian Blythe and Parker Bruni
# CS 434
# Implementation Assignment 2
# Problem 2

import numpy as np
import random
import math

#format output to be more readable
np.set_printoptions(suppress=True)

#----------------------------------------
#CLASSES
#----------------------------------------

class Node():
    def __init__(self):
        self.parent = None
        self.leftchild = None
        self.rightchild = None
        self.list = None
        self.output = None
        self.threshold = None
        self.feature = None
    
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

    threshold = sortedarray[idx_max_splt,:][0,1]
    
    return posL, negL, posR, negR, idx_max_splt, maxgain, threshold


#------------------------------------------
# Function: infoGain
# Description: calculates the information gain
#------------------------------------------

def infoGain(index, numrows, entropyM, entropyL, entropyR):

    gain = entropyM - (float(index)/numrows*entropyL) - ((1-float(index)/numrows)*entropyR)

    return gain

#------------------------------------------
# Function: allsame
# Description: true if all values in array are the same
#              false otherwise
#------------------------------------------

def allsame(arr):

    if np.all(arr == 1) or np.all(arr == -1):
        return True
    else:
        return False

#------------------------------------------
# Function: findlocation
# Description: compares the features with runninglist and pick best
#------------------------------------------

def findlocation(arr, runninglist):

    sort = arr[arr[:, 2].argsort()]

    i = 0
    condition = True
    while condition == True:
        location = int(sort[-i-1][0]) # set location max value then iterate down if already used
        if location in runninglist:
            i = i+1
        else:
            condition = False
            runninglist = runninglist + [location]

    return runninglist, location
    
#------------------------------------------
# Function: decistionTree
# Description: Runs the decision Tree algorithm for a single stump
#------------------------------------------

def makeDecisionTree(currentnode, d, runninglist):
    '''
    cycle through all versions with the top node of the decision tree
    being a new column of data
    '''

    print currentnode.output.T

    if d == 0: #base case
        print "base case met"
        return

    #checks if the current list is all one type
    if allsame(currentnode.output):
        print "Node all one type"
        return 

    X = currentnode.list
    Y = currentnode.output

    totals = []

    numcols = X.shape[1]
    print "numcols: ", numcols

    print "Depths to expand: ", d 

    for i in range(numcols): 
        posL, negL, posR, negR, idx_max_splt, maxgain, threshold = calculateSplit(X[:,i], Y) 
        totals = totals + [[i, idx_max_splt, maxgain, threshold]] #add values to list

    arr = np.array(totals) #make into numpy array

    maxindex, maxsplt, maxgain, maxthresh = arr.max(axis=0) 
    runninglist, location = findlocation(arr, runninglist)

    #location = arr.argmax(axis=0)[2]
    print "Feature number: ", location
    print "Split at index: ", arr[location,1]
    print "index threshold: ", arr[location,3]
    print "Gain of division: ", maxgain

    #set up new inputs
    XplusY = np.concatenate((X, Y), axis = 1)

    sortedX = XplusY[np.lexsort(np.fliplr(XplusY).T[numcols-location])][0,:,:] #sort by chosen parameter
    #sortedX = np.delete(sortedX, location, 1) #delete used parameter
    sortedXL, sortedXR = sortedX[:int(arr[location,1]),:], sortedX[int(arr[location,1]):,:] #binary split by index


    newYL = sortedXL[:,-1]
    newYR = sortedXR[:,-1]
    
    sortedXL = sortedXL[:,:-1] #get rid of last column (Y)
    sortedXR = sortedXR[:,:-1] #get rid of last column (Y)

    leftnode = Node()
    rightnode = Node()

    currentnode.threshold = arr[location,3]
    currentnode.feature = location
    
    currentnode.leftchild = leftnode
    currentnode.rightchild = rightnode
    rightnode.parent = currentnode
    leftnode.parent = currentnode
    leftnode.list = sortedXL
    rightnode.list = sortedXR
    leftnode.output = newYL
    rightnode.output = newYR

    #print leftnode.list
    #print leftnode.output
    print "\n"
    print "entering left node"
    print "shape of list: ", leftnode.list.shape
    makeDecisionTree(leftnode, d-1, runninglist)

    print "\n"
    print "entering right node"
    print "shape of list: ", rightnode.list.shape
    makeDecisionTree(rightnode, d-1, runninglist)
        
#------------------------------------------
# Function: testDecistionTree
# Description: runs test data through the decision tree
#------------------------------------------

def testDecisionTree(trainnode, testnode, d):

    print "Depths to expand: ", d
    print testnode.output.T

    if d == 0: #base case
        print "base case met"
        return

    if testnode.list.size == 0:
        print "empty list"
        return

    #checks if the current list is all one type
    if trainnode.leftchild == None and trainnode.rightchild == None:
        print "Reached training leaf"
        print 
        return 

    thresh = trainnode.threshold
    feat = trainnode.feature

    print "thresh: ", thresh
    print "feat: ", feat

    numcols = testnode.list.shape[1]

    XplusY = np.concatenate((testnode.list, testnode.output), axis = 1) #combine test input and output
    sortedX = XplusY[np.lexsort(np.fliplr(XplusY).T[numcols-feat])][0,:,:] #sort test by trained parameter (feat)

    splitloc = np.argmax(sortedX[:,feat]>thresh) #find split where all values are above threshold
    #sortedX = np.delete(sortedX, feat, 1) #delete used parameter

    sortedXL, sortedXR = sortedX[:splitloc,:], sortedX[splitloc:,:] #binary split by index
    

    newYL = sortedXL[:,-1]
    newYR = sortedXR[:,-1]

    sortedXL = sortedXL[:,:-1] #get rid of last column (Y)
    sortedXR = sortedXR[:,:-1] #get rid of last column (Y)

    leftnode = Node()
    rightnode = Node()

    testnode.leftchild = leftnode
    testnode.rightchild = rightnode
    rightnode.parent = testnode
    leftnode.parent = testnode
    leftnode.list = sortedXL
    rightnode.list = sortedXR
    leftnode.output = newYL
    rightnode.output = newYR
    
    print "\n"
    print "entering left node"
    print "shape of list: ", leftnode.list.shape
    testDecisionTree(trainnode.leftchild, leftnode, d-1)

    print "\n"
    print "entering right node"
    print "shape of list: ", rightnode.list.shape
    testDecisionTree(trainnode.rightchild, rightnode, d-1)

#------------------------------------------
# Function: errorStump
# Description: gets the error of the built tree's stump
#------------------------------------------

def errorStump(rootnode):
	correctGuess = 0.0
	
	#for every value in the left nodes output array, keep track of correct guesses
	for i in range(len(rootnode.leftchild.output)):
		if (rootnode.leftchild.output[i] == -1):
			correctGuess = correctGuess + 1.0
	#for every value in the right nodes output array, keep track of correct guesses		
	for i in range(len(rootnode.rightchild.output)):
		if (rootnode.rightchild.output[i] == 1):
			correctGuess = correctGuess + 1.0
	
	#error = 1 - the number of correct guesses divided by the total number of elements in the list.
	error = (1.0 - (correctGuess / len(rootnode.list)))
	
	return error


#------------------------------------------
# Function: errorTree
# Description: gets the error of the entire build tree
#------------------------------------------

def errorTree(rootnode):
	correctGuess = 0.0
		
	correctGuess = leftLeavesCorrectGuesses(rootnode) + rightLeavesCorrectGuesses(rootnode)
			
	#error = 1 - the number of correct guesses divided by the total number of elements in both the left and right lists.
	error = (1.0 - (correctGuess / len(rootnode.list)))
	
	return error
	

#------------------------------------------
# Function: isLeaf
# Description: determines if a node is a leaf
#------------------------------------------
def isLeaf(node):
	if node is None:
		return False
	if node.leftchild is None and node.rightchild is None:
		return True
	return False
	
#------------------------------------------
# Function: leftLeavesCorrectGuesses
# Description: gets the number of correct guesses in the left-sided leaves, which expects all values to be -1
#------------------------------------------
def leftLeavesCorrectGuesses(node):
	# Initialize result
    correctGuess = 0
     
    # Update result if root is not None
    if node:
        # If left child is None, it is a leaf, get its data
        if isLeaf(node.leftchild):
            for i in range(len(node.leftchild.output)):
				if (node.leftchild.output[i] == -1):
					correctGuess = correctGuess + 1.0
        else:
            # Else recur for left child of root
            correctGuess += leftLeavesCorrectGuesses(node.leftchild)
 
        # Recur for right child of root and update CorrectGuess
        correctGuess += leftLeavesCorrectGuesses(node.rightchild)
    return correctGuess

	
#------------------------------------------
# Function: rightLeavesCorrectGuesses
# Description: gets the number of correct guesses in the right-sided leaves, which expects all values to be 1
#------------------------------------------
def rightLeavesCorrectGuesses(node):
	# Initialize result
    correctGuess = 0
     
    # Update result if root is not None
    if node:
        # If right child is None, it is a leaf, get its data
        if isLeaf(node.rightchild):
            for i in range(len(node.rightchild.output)):
				if (node.rightchild.output[i] == 1):
					correctGuess = correctGuess + 1.0
        else:
            # Else recur for right child of root
            correctGuess += rightLeavesCorrectGuesses(node.rightchild)
 
        # Recur for left child of root and update correctGuess
        correctGuess += rightLeavesCorrectGuesses(node.leftchild)
    return correctGuess

#-------------------------------------------
# Function: MAIN
#-------------------------------------------

X, Xtest, Y, Ytest = loadData()

d = 6

rootnode = Node() #set up root of the tree
rootnode.list = X
rootnode.output = Y
runninglist = []

print "making decision tree..."
makeDecisionTree(rootnode, d, runninglist)

testroot = Node()
testroot.list = Xtest
testroot.output = Ytest

print "making test tree..."
testDecisionTree(rootnode, testroot, d)

print "Training Data"
print "Stump error: ", errorStump(rootnode)
print "Tree of size" , d, "error: ",errorTree(rootnode)

print "Testing Data"
print "Stump error: ", errorStump(testroot)
print "Tree of size" , d, "error: ",errorTree(testroot)





