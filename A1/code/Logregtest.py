import numpy

#pick these values
epsilon = 13
alpha = 0.001

#format output to be more readable
numpy.set_printoptions(suppress=True)

#load all the training data into an array X
Xdata = numpy.genfromtxt ('usps-4-9-train.csv', delimiter=",")
X = numpy.matrix(Xdata[:, :-1])

#load all the testing data into an array Xtest
Xtestdata = numpy.genfromtxt ('usps-4-9-test.csv', delimiter=",")
Xtest = numpy.matrix(Xtestdata[:, :-1])

#set Y array to the last column of X data, transpose Y 
Y = numpy.matrix(Xdata[:,-1]).T
Ytest = numpy.matrix(Xtestdata[:,-1]).T

Yt = numpy.transpose(Y)

#divide features by 255 to avoid overflow
X = X/255
Xt = numpy.transpose(X)

Xtest = Xtest/255

#define sigmoid fucntions
def sigmoid(x):
    return 1.0 / (1.0 + numpy.exp(-numpy.asscalar(x)))

def sigmoid_array(x):
    return 1.0 / (1.0 + numpy.exp(-x))


#initialize guess for w (guess all zeros)
w = numpy.zeros((1, X.shape[1]))

#Run iterative gradient descent
#for j in range(max_iterations):
ftrain = open('trainOutput.csv', 'w+')
ftest = open('testOutput.csv', 'w+')
count = 0
while True:

    #initialize grad
	grad = numpy.array([0] * X.shape[1], 'float128')
	#needed to calculate how many correct predictions were made
	testcorrect = 0
	traincorrect = 0
	testpercentage = 0
	trainpercentage = 0
	
	for i in range(X.shape[0]):
	
		Xi = X[i,:] #take each row of input
		Yi = Y[i,:] #take each target output
           
		yhat = sigmoid(Xi * w.T) #compute sigmoid of w.T multiplied by input row

		grad = grad + (yhat - Yi)*Xi #compute grad difference 
	
	w = w - (alpha * grad) 
	count = count + 1

	#apply the model to the training and test data to keep track of how it evolves
	P = sigmoid_array(numpy.matmul(X, w.T))
	Ptest = sigmoid_array(numpy.matmul(Xtest, w.T))
	
	#calculate if the value was correctly predicted for both test data and train data
	for k in range(P.shape[0]):
		if (P[k,:] >= .5 and Y[k,:] == 1) or (P[k,:] < .5 and Y[k,:] == 0):
			testcorrect = testcorrect + 1
			
	for k in range(Ptest.shape[0]):
		if (Ptest[k,:] >= .5 and Ytest[k,:] == 1) or (Ptest[k,:] < .5 and Ytest[k,:] == 0):
			traincorrect = traincorrect + 1
	
	
	testpercentage = testcorrect / 1400.0
	trainpercentage = traincorrect / 800.0
	
	ftrain.write(str(testpercentage) + "\n")
	ftest.write(str(trainpercentage) + "\n")
	
    
	if numpy.linalg.norm(grad) <= epsilon or count >= 100:
		break
ftrain.close()
ftest.close()
print w
print count

