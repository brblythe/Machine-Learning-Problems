# Parker Bruni and Brian Blythe
# Final Programming Assignment
# Final mini project/competition


from copy import deepcopy
import numpy as np

class window:

	def __init__(self):
		self.features = np.zeros(7,9) #define the window size (7 rows)
		self.status = 0 #set to 1 if hypoglycemic

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
	chunks_arr = []
	
	for i in range(data.shape[0]):	#for each row of the data
		if (data.shape[0] - i) >= 7: #if there are still 7 rows before end of file
			chunk = window() #create a new, 7 row, window 
			for j in range(7):
				chunk.features[j] = np.asarray(data[i+j]) #assign all 7 rows to this window
			
			chunk.status = data[i+6][8]
		chunks_arr = np.append(chunks_arr, chunk, axis=0)	#push the new row chunk to the chunks array
	
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