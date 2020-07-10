# Parker Bruni and Brian Blythe
# Final Programming Assignment
# Final mini project/competition


from copy import deepcopy
import numpy as np

#--------------------------------------------
# Function: importdata
# Description: imports relevant data
#--------------------------------------------

def importdata(file):

    data = np.loadtxt(file, delimiter = ',', usecols=range(1,10))

    return data


#--------------------------------------------
# Function: parseChunks
# Description: puts 30 minutes of feature data into 1 chunk
#--------------------------------------------    

def parseChunks(data):
	chunks_arr = np.zeros((1,9))
	print chunks_arr.shape
	for x in range(data.shape[0]):				#for each row in the raw data, create a new row with features from the next 7 rows of features
		chunk = np.matrix(data[x])						#initialize first row of chunk
		for y in range(1,7):					#for the current index of x, get combine the next 6 rows of features (excluding the already initialized first row)
			if (data.shape[0] - x) >= 7:		#if there is still at least 7 entries ahead of the current index point in the data
				next_row = np.matrix(data[x+y])
				chunk = np.concatenate((chunk, next_row), axis=0)
		chunks_arr = np.append(chunks_arr, chunk, axis=0)	#push the new row chunk to the chunks array
	print chunks_arr
	return chunks_arr
#--------------------------------------------
# Function: MAIN
#--------------------------------------------

#import data
data1 = importdata('Subject_1.csv')
data2 = importdata('Subject_4.csv')
data3 = importdata('Subject_6.csv')
data4 = importdata('Subject_9.csv')

#concatenate data for general case model
data_all = np.concatenate((data1, data2), axis=0)
data_all = np.concatenate((data_all, data3), axis=0)
data_all = np.concatenate((data_all, data4), axis=0)

parseChunks(data_all)