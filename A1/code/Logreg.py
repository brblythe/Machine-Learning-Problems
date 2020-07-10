import numpy

#pick these values
epsilon = 15
alpha = 0.001
lmda = 0.0005

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

count = 0

while False:

    #initialize grad
    grad = numpy.array([0] * X.shape[1], 'float128')

    for i in range(X.shape[0]):

        Xi = X[i,:] #take each row of input
        Yi = Y[i,:] #take each target output
           
        yhat = sigmoid(Xi * w.T) #compute sigmoid of w.T multiplied by input row

        grad = grad + (yhat - Yi)*Xi #compute grad difference 

    w = w - (alpha * grad) #compute weight vector
    count = count + 1

    P = sigmoid_array(numpy.matmul(Xtest, w.T))
    
    
    if numpy.linalg.norm(grad) <= epsilon or count > 200:
        break

print w
print count

#run logistic regression with regularization

#initialize guess for w (guess all zeros)
w = numpy.zeros((1, X.shape[1]))

#Run iterative gradient descent

count = 0

while True:

    #initialize grad
    grad = numpy.array([0] * X.shape[1], 'float128')

    for j in range(X.shape[0]):

        Xi = X[j,:] #take each row of input
        Yi = Y[j,:] #take each target output
           
        yhat = sigmoid(Xi * w.T) #compute sigmoid of w.T multiplied by input row

        norm = 2 * w

        grad = grad + (yhat - Yi)*Xi + (0.5 * lmda * norm) #compute grad difference 

    w = w - (alpha * grad) #compute weight vector
    count = count + 1

    P = sigmoid_array(numpy.matmul(Xtest, w.T))
    
    
    if numpy.linalg.norm(grad) <= epsilon or count > 200:
        break

print w
print count

