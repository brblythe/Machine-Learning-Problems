# Parker Bruni and Brian Blythe
# Programming Assignment 4
# K Means
# K means code was produced with help from code found here:
# https://mubaris.com/2017/10/01/kmeans-clustering-in-python/

from copy import deepcopy
import numpy as np

#--------------------------------------------
# Function: importdata
# Description: imports relevant data
#--------------------------------------------

def importdata(file):

    data = np.loadtxt(file, delimiter = ',')

    return data

#--------------------------------------------
# Function: initseeds
# Description: initialize k random cluster centers
#--------------------------------------------

def initseeds(k, datashape):

    numfeatures = datashape[1]
    seeds = np.ones([k, numfeatures])
    for i in range(k):
        randomseed = [(255*np.random.rand(numfeatures)).astype(int)][0]
        seeds[i,:] = randomseed

    return seeds

#--------------------------------------------
# Function: eucdist
# Description: Computes the euclidean distance
#              between two points
#--------------------------------------------

def eucdist(a, b, ax=1):
    
    dist = np.linalg.norm(a-b, axis = ax)

    return dist

#--------------------------------------------
# Function: sse
# Description: Computes the sum of squared error
#              of two arrays
#--------------------------------------------

def sse(a, b, ax=None):
 
    sser = ((a - b) ** 2).sum(axis=ax)

    return sser

#--------------------------------------------
# Function: sse_all
# Description: Computes the sum of squared error
#              of two arrays
#--------------------------------------------

def sse_all(seeds, data, clusters, ax=None):

    total_sse = 0
    
    for j in range(k): #for each seed
        for i in range(len(data)): #loop for each data point
            row_sse = 0
            if clusters[i] == j: #if that data point is associated with that seed
                row_sse = sse(data[i], seeds[j]) #calculate the sse of that data point from the seed

            total_sse = total_sse + row_sse
        
    return total_sse

#--------------------------------------------
# Function: kmeans
# Description:
#
#--------------------------------------------

def kmeans(data, seeds, k):

    print 'k: ', k

    #to store the values of the seeds
    seeds_old = np.zeros(seeds.shape)

    #to store cluster assignments
    clusters = np.zeros(len(data))

    #initialize sse error for loop
    error = sse(seeds, seeds_old)
    print 'Initial seed error: ', error

    #will run until error is 0, i.e. the function converges
    while error != 0:

        for i in range(len(data)): #for each point (row) of data
            #compute the distance from it to each seed
            distances = eucdist(data[i], seeds)
            #find the nearest cluster index
            cluster = np.argmin(distances)
            #associate the data point with a cluster
            clusters[i] = cluster 

        seeds_old = deepcopy(seeds) #store the previous seed location

        
        #find the new seed locations

        for i in range(k): #for each seed

            x = 0
            points = np.zeros(data.shape)#(initialize points list sufficiently large)
            
            for j in range(len(data)): #and for every point
                
                if clusters[j] == i: #if the point is associated with that seed

                    points[x] = data[j] # add that point to the points array
                    x = x + 1 #and increment points to next slot


            points = points[0:x,:] #remove trailing zeros

            if x == 0: #if the seed is not the closest to any points, leave it alone
                pass
            else:
                seeds[i] = np.mean(points, axis=0) #otherwise set seeds to mean of assigned points
        
        error = sse(seeds, seeds_old) #assigns 0 if converged
            
        #find sse of all clusters to all seeds
        total_sse = sse_all(seeds, data, clusters)

        print 'total sse: ', total_sse

        #reference global minSSE
        global minSSE
		
        if total_sse < minSSE or minSSE == 0:
            minSSE = total_sse

        if error == 0:
            print 'converged'
    
    
#--------------------------------------------
# Function: MAIN
#--------------------------------------------

k = 10
print k
data = importdata('data-1.txt')
minSSE = 0

for i in range(10):
    seeds = initseeds(k, data.shape)
    kmeans(data, seeds, k)

output = open('kmeansOutput.csv','w')
output.write("Minimum SSE\n")
output.write("%.2i\n" % (minSSE))
output.close() 
