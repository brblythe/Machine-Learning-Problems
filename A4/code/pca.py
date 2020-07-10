# Parker Bruni and Brian Blythe
# Programming Assignment 4
# Principle Component Analysis
# Principle Component Analysis code was produced with help from code found here:
# https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/



import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
from os import mkdir
from os.path import isdir, join
PLOT_DIR = 'pca_plots'
SAVE_PLOT = False



#--------------------------------------------
# Function: importdata
# Description: imports relevant data
#--------------------------------------------

def importdata(file):

    data = np.loadtxt(file, delimiter = ',')

    return data
	

#--------------------------------------------
# Function: MAIN
#--------------------------------------------

data = importdata('data-1.txt')
#print(data)

# calculate the mean of each column
M = mean(data.T, axis=1)
#print M

# center columns by subtracting column means
C = data - M
#print C

# calculate covariance matrix of centered matrix
V = cov(C.T)
#print V

# eigen decomposition of covariance matrix
values, vectors = eig(V)
#print vectors
print values[:10]

# project data
P = vectors.T.dot(C.T)
#print P.T

# remove imaginary parts of eigen values to prevent error message on output file write
values = values.real

output = open('pcaOutput.csv','w')
for i in range(10):
	output.write("%.2f\n" % (values[i])) 
output.close() 



################# Part 3.2 ################################################
if not isdir(PLOT_DIR) and SAVE_PLOT:
        mkdir(PLOT_DIR)

columns = 2
rows = 6
fig, axarr = plt.subplots(rows, columns, figsize=(10, 10))
fig.suptitle("Mean image and Top Ten Eigenvectors")
axarr[0][0].imshow(pca.M.values.reshape(28, 28), cmap='gray')
axarr[0][0].axis('off')
axarr[0][0].set_title("Mean Image")
axarr[0][1].axis('off')
for j in range(5):
	for i in range(2):
		eig_num = 2 * (j) + i
		img = pca.principle_cpts.iloc[eig_num, :].values.reshape(28, 28)
		# Skip first row (first column is set to have mean image)
		axarr[j + 1, i].imshow(img, cmap='gray')
		axarr[j + 1, i].set_title(("Eigenvector " + str(eig_num + 1)))
		axarr[j + 1, i].axis('off')

if SAVE_PLOT:
	plt.savefig(join(PLOT_DIR, 'plot3_2.eps'))
	plt.savefig(join(PLOT_DIR, 'plot3_2.png'))
plt.show()

###############################################################################




