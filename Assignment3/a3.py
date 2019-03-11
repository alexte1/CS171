import matplotlib.pyplot as plt
import random #this is so we can generate k random numbers to choose for init_centroids
import numpy as np

def Sensitivity(data, assignments, centroids, k):

	meanList = []
	stdList = []
	for i in range(k):
		tmp = []
		for j in range(len(assignments)):
			if assignments[j] == i:
				tmp.append(data[j])
		meanList.append(np.mean(tmp))
		stdList.append(np.std(tmp))

	print "Mean List - ", meanList

	return (meanList, stdList)

def SSE(k, assignments, centroids, data_points):
	SSE_List = []
	stdList = []
	# print len(centroids)

	for i in range(k):
		tmp = []
		for j in range(len(assignments)):
			if assignments[j] == i:
				square_Error = distance(data_points[j], centroids[i])
				# **2 because we need to square it since sum of square error is (p - c)^2
				square_Error = square_Error**2
				tmp.append(square_Error)
		SSE_List.append(tmp)
		stdList.append(np.std(tmp))
	lis = []
	for i in range(k):
		tmper = 0.0
		for j in range(len(SSE_List[i])):
			tmper += SSE_List[i][j]

		lis.append(tmper)

	value = np.sum(lis)
	return value

	
def scatter_plot(data, cluster_assignments, cluster_centroids, k):
	colors = ['r', 'g', 'b', 'y', 'c', 'm']
	fig, ax = plt.subplots()
	for i in range(k):
	        points = np.array([data[j] for j in range(len(data)) if cluster_assignments[j] == i])
	        ax.scatter(points[:,2], points[:,3], s=7, c=colors[i], alpha=0.3)
	ax.scatter(cluster_centroids[:,2], cluster_centroids[:,3], marker='*', s=200, c='#050505')
	# plt.savefig('plots/scatterplot', bbox_inches='tight')
	plt.show()

def distance(x,y):
	#exclusively using euclidean distance (p = 2).
	retSum = 0.0
	for i,j in zip(x,y):
		retSum += abs(i-j)**2
	return retSum**(1./2)

def k_means2(x_input, k, init_centroids):

	number_of_runs = 0
	newCentroids = []
	while np.any(init_centroids != newCentroids):
		distanceArray = []
		clusterNumber = []
		dataInCluster = [[0,0,0,0]] * k
		number_of_runs += 1
		newCentroids = []
		#computing the data and clustering the data
		for i in range(len(x_input)):
			minDist = 99999
			for j in range(k):
				dist = distance(x_input[i], init_centroids[j])
				if dist < minDist:
					minDist = dist
					tmp = j
			#stores each distance so that we can use for SSE
			distanceArray.append(minDist)
			clusterNumber.append(tmp)
			if dataInCluster[tmp] == [0,0,0,0]:
				dataInCluster[tmp] = [x_input[i]]
			else:
				dataInCluster[tmp].append(x_input[i])
		#calculates the sum and divides it by the length (gets the mean)
		for i in range(k):
			tmpList = [0,0,0,0]
			if len(dataInCluster[i]) != 0:
				for j in range(len(dataInCluster[i])):
					#for a in range(len(dataInCluster[i][j])):
					for a in range(len(tmpList)):
						#checks the bounds
						if dataInCluster[i][j] != 0:
							#print "Data in cluster - ", dataInCluster[i][j][a]
							tmpList[a] = tmpList[a] + dataInCluster[i][j][a]
				tmp = []
				# print tmpList
				for c in range(4):
					tmp.append(tmpList[c] / len(dataInCluster[i]))
				newCentroids.append(tmp)
			if len(dataInCluster[i]) != 0:
				init_centroids = newCentroids

	#print "Iterations - ", number_of_runs

	# print len(dataInCluster[0])
	# print len(dataInCluster[1])

	# temper = []
	# zero = 0
	# one = 0
	# two = 0
	# for i in range(len(clusterNumber)):
	# 	if clusterNumber[i] == 0:
	# 		zero += 1
	# 	elif clusterNumber[i] == 1:
	# 		one += 1
	# 	else:
	# 		two += 1
	# print "Zeros - ", zero
	# print "Ones - ", one
	# print "Two - ", two

	return (clusterNumber, newCentroids)

def main():

	print "TAKE OUT THE SCATTERPLOT"

	irisFile = open("iris", "r")

	unparsedirisData = []
	parsedData = []
	#	stores the name of the flower for each of the data points. (all 150), then deletes them.
	flowerName = []
	unparsedirisData = irisFile.readlines()
	tmp = []
	####################################################################

	#Handles the parsing in and taking out the '\n' after each readline#

	for index in unparsedirisData:
		parsedData.append(index.split(","))
	for index in parsedData:
		index[-1] = index[-1].strip();			## <<<<< this is the function "strip" that takes out the \n
 	del parsedData[-1]

	#	stores the name of the flower for each of the data points. (all 150), then deletes them.
	for i in range(len(parsedData)):
		flowerName.append(parsedData[i][-1])
		del parsedData[i][-1]

	for i in range(len(parsedData)):
		for j in range(len(parsedData[i])):
			parsedData[i][j] = float(parsedData[i][j])
	# ####################################################################
	irisFile.close()

	#goes from k = 1 to 10

	#the y axis value from SSE call.
	y_values = []
	x_values = [1,2,3,4,5,6,7,8,9,10]

	for k in range(1, 11):
			#	"Chooses" k random numbers between 0 (inclusive) and len(irisData) + 1 (exclusive).
			#init_centroids = random.sample(range(0,len(parsedData) + 1), k)
		init_centroids = random.sample(range(0,len(parsedData)), k)

		#	sorta extra, but instead of creating a tmp array, just take the list returned from random.sample (which hold the two random values/indexes) and find the data x feature for init_centroids[0 - k]
		for i in range(k):
			init_centroids[i] = parsedData[init_centroids[i]]


		#tupl is a tuple that is returned by k_means function. First index holds cluster_assignments. Second index holds cluster_centroids.
		assignments, centroids = k_means2(parsedData,k,init_centroids)

		y_values.append(SSE(k, assignments, centroids, parsedData))

	plt.plot(x_values, y_values)
	plt.show()

	#########################################################################################################
	#for sensitivity 

	y_values = []
	y2_values = []
	error_values = []
	max_iter = [2,10,100]
	for a in range(len(max_iter)):
		for k in range(1, 11):
			for b in range(max_iter[a]):
				#	"Chooses" k random numbers between 0 (inclusive) and len(irisData) + 1 (exclusive).
				#init_centroids = random.sample(range(0,len(parsedData) + 1), k)
				init_centroids = random.sample(range(0,len(parsedData)), k)

				#	sorta extra, but instead of creating a tmp array, just take the list returned from random.sample (which hold the two random values/indexes) and find the data x feature for init_centroids[0 - k]
				for i in range(k):
					init_centroids[i] = parsedData[init_centroids[i]]


				#tupl is a tuple that is returned by k_means function. First index holds cluster_assignments. Second index holds cluster_centroids.
				assignments, centroids = k_means2(parsedData,k,init_centroids)
				xyz = SSE(k, assignments, centroids, parsedData)
				print "A - ", a
				print "K - ", k
				print "B - ", b
				print "Storing - ", xyz
				y_values.append(xyz)
			print "Sum - ", np.sum(y_values)
			print "max_iter[a] - ", max_iter[a]
			print "Std - ", np.std(y_values)
			tmp = np.sum(y_values) / max_iter[a]
			#tmp = np.sum(y_values)/ len(parsedData)
			error_values.append(np.std(y_values))
			y_values = []
			y2_values.append(tmp)
		# plt.plot(x_values, y2_values)
		plt.errorbar(x_values, y2_values, yerr = error_values)
		y2_values = []
		error_values = []
		plt.show()
		
	
	#scatter_plot(parsedData,assignments, np.array(cluster_centroids), k)

main()