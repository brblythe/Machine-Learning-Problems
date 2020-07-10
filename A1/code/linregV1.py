import numpy
#format output to be more readable
numpy.set_printoptions(suppress=True)

#load all the training data into an array X
Xdata = numpy.loadtxt("housing_train.txt", dtype=float)
X = numpy.matrix(Xdata[:, :-1])

#load all the testing data into an array
Xtestdata = numpy.loadtxt("housing_test.txt", dtype=float)
Xtest = numpy.matrix(Xtestdata[:, :-1])

#set Y array to the last column of X data, transpose Y 
Y = numpy.matrix(Xdata[:,-1]).T
Ytest = numpy.matrix(Xtestdata[:,-1]).T

#insert a dummy variable "1" at the beginning of all rows
X = numpy.hstack((numpy.ones((X.shape[0], 1)), X))
Xtest = numpy.hstack((numpy.ones((Xtest.shape[0], 1)), Xtest))

Xt = X.T

# compute the weight vector w
a = (numpy.matmul(Xt,X))**-1
b = numpy.matmul(Xt,Y)
w = numpy.matmul(a,b)

#format w to be more readable
wprint = numpy.round(w,3)
print '1.1'
print 'weight vector w \n', wprint

#generate ASE for training and test data

traindiff = (Y - numpy.matmul(X,w))
SSEtrain = numpy.matmul(numpy.transpose(traindiff),(traindiff))
ASEtrain = SSEtrain / X.shape[0]

testdiff = (Ytest - numpy.matmul(Xtest,w))
SSEtest = numpy.matmul(numpy.transpose(testdiff),(testdiff))
ASEtest = SSEtest / Xtest.shape[0]

print '1.2'
print 'ASE for trained data: ', ASEtrain
print 'ASE for test data: ', ASEtest

#generate SSE and ASE for training and test data without dummy variable added.

X2 = numpy.delete(X, 0, 1)
X2t = numpy.transpose(X2)

X2test = numpy.delete(Xtest, 0, 1)

# compute the weight vector w
a = (numpy.matmul(X2t,X2))**-1
b = numpy.matmul(X2t,Y)
w2 = numpy.matmul(a,b)

wprint = numpy.round(w2,3)
print '1.3'
print 'weight vector w \n', wprint

#generate ASE for training and test data

traindiff = (Y - numpy.matmul(X2,w2))
SSEtrain = numpy.matmul(numpy.transpose(traindiff),(traindiff))
ASEtrain = SSEtrain / X2.shape[0]

testdiff = (Ytest - numpy.matmul(X2test,w2))
SSEtest = numpy.matmul(numpy.transpose(testdiff),(testdiff))
ASEtest = SSEtest / X2test.shape[0]

print 'ASE for trained data: ', ASEtrain
print 'ASE for test data: ', ASEtest

#Modify data with random features

numfeat = int(input("How many additional features would you like to add? "))

Xrand = numpy.c_[X, numpy.random.randn(X.shape[0], numfeat)]
Xrandt = numpy.transpose(Xrand)
Xtestrand = numpy.c_[Xtest, numpy.random.randn(Xtest.shape[0], numfeat)]

# compute the weight vector w
a = (numpy.matmul(Xrandt,Xrand))**-1
b = numpy.matmul(Xrandt,Y)
w = numpy.matmul(a,b)

#generate ASE for training and test data

traindiff = (Y - numpy.matmul(Xrand,w))
SSEtrain = numpy.matmul(numpy.transpose(traindiff),(traindiff))
ASEtrain = SSEtrain / Xrand.shape[0]

testdiff = (Ytest - numpy.matmul(Xtestrand,w))
SSEtest = numpy.matmul(numpy.transpose(testdiff),(testdiff))
ASEtest = SSEtest / Xtest.shape[0]

print '1.4'
print 'ASE for trained data: ', ASEtrain
print 'ASE for test data: ', ASEtest

