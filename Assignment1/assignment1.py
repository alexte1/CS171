import matplotlib.pyplot as plt
import numpy as np
import math

# def distance(x, y, p):



def mean(x):

	val = 0

	for index in range(len(x)):
		val += x[index]
	return val / (len(x) - 1)
def stdev(x):

	tmp = []
	mean2 = mean(x)
	val = 0

	for index in range(len(x)):
		tmp.append((x[index] - mean2)**2)
	for index in range(len(tmp)):
		val += tmp[index] 
	return math.sqrt(val / (len(x) - 1))
def Pearson(x, y, z):

	return x / (y * z)
def covariance(x,y):
	cov1 = []
	cov2 = []
	xMean = mean(x)
	yMean = mean(y)
	covarianceArray = []

	for index in range(len(x)):
		cov1.append(x[index] - xMean)

	for index in range(len(y)):
		cov2.append(y[index] - yMean)

	for x1,y1 in zip(cov1,cov2):
		covarianceArray.append(x1 * y1)

	return np.mean(covarianceArray)
def correlation(dataset):

	# for index in dataset:
	# 	print dataset

	#dataset = dataset[1:]

	# for index in dataset:
	# 	print dataset

	tmpArray = []
	rtnArray = []
	tmpArray = np.transpose(np.asarray(dataset))

	# for index in tmpArray:
	# 	print index

	for feature1 in tmpArray:
		for feature2 in tmpArray:	
			rtnArray.append(Pearson(covariance(feature1,feature2), stdev(feature1), stdev(feature2)))

	return rtnArray
def heatMap(dataset):
	dataset = np.reshape(np.asarray(dataset), (-1,13))
	fig = plt.figure()
	# x = plt.matshow(dataset)
	ax = fig.add_subplot(111)
	cax = ax.matshow(dataset, cmap="YlOrRd")
	for i in xrange(13):
		for x in xrange(13):
			c = dataset[x, i]
			#plt.text(i, x, str(c), va = 'center', ha='center')
	plt.colorbar(cax)
	plt.show()
def scatterPlot(dataset):

	dataset = dataset[:-1]

	colorArray = []
	for index in range(len(dataset)):
		if index < 50:
			colorArray.append('b')
		elif index < 99:
			colorArray.append('r')
		else:
			colorArray.append('g')
	tmpArray = []

	tmpArray = np.transpose(np.asarray(dataset))

	for feature1 in range(len(tmpArray)):
		for feature2 in range(len(tmpArray)):
			plt.figure()
			plt.scatter(tmpArray[feature1], tmpArray[feature2],c = colorArray, alpha = 0.2)
			#plt.show()
			plt.savefig("irisscatterplots/" + str(feature1) + '_' + str(feature2) + ".png", bbox_inches='tight')
def main():
	while 1:
		binNum = input("Enter number of bins - ")
		#open the specific files and put the delimiter as "r" so that it is in read mode.
		irisFile = open("iris", "r")
		wineFile = open("wine", "r")

		i = 0
		j = 0
		k = 0
		l = 0

		#IRIS DATASET
		irisSetosaArray = []
		irisVersiColorArray = []
		irisVirginicaArray = []
		allThreeIrisArray = []

		#WINE DATASET
		class1 = []
		class2 = []
		class3 = []
		allThreeWines = []
		intwineArray = []

		#get each class sepearated for the Wine Data Set
		while i < 178 and j < 3:
			wineArray = wineFile.readline()
			wineArray = wineArray[:-1]
			if j == 0:
				class1.append(wineArray)
			if j == 1:
				class2.append(wineArray)
			if j == 2:
				class3.append(wineArray)

			i += 1
			if i >= 48:
				if i == 59 and j == 0:
					i = 0
					j += 1
				if i == 71 and j == 1:
					i = 0
					j+= 1
				if i == 48 and j == 2:
					i = 0
					j += 1

		#get each class seperated for the Iris Data Set
		while k < 50 and l < 3:
			irisArray = irisFile.readline()
			if l == 0:
				irisSetosaArray.append(irisArray)
			if l == 1:
				irisVersiColorArray.append(irisArray)
			if l == 2:
				irisVirginicaArray.append(irisArray)

			k += 1
			if k == 50:
				k = 0
				l += 1

		#case where the user wants to read data from all 3 in wine and iris
		irisFile2 = open("iris", "r")
		allThreeIrisArray = irisFile2.readlines()
		wineFile2 = open("wine", "r")
		allThreeWines = wineFile2.readlines()

		#IRIS DATASET ARRAYS
		Parsed_irisSetosaArray = []
		Parsed_irisVersiColorArray = []
		Parsed_irisVirginicaArray = []
		Parsed_allThree = []
		Parsed_allThree2 = []

		#WINE DATASET ARRAYS
		Parsed_class1 = []
		Parsed_class2 = []
		Parsed_class3 = []
		Parsed_allThreeWine = []
		Parsed_allThreeWine2 = []

		#parse each class in Wine
		for index in class1:
			parsed = index.split(",")
			parsed = [float(i) for i in parsed]
			Parsed_class1.append(parsed)
		for index in class2:
			parsed = index.split(",")
			parsed = [float(i) for i in parsed]
			Parsed_class2.append(parsed)
		for index in class3:
			parsed = index.split(",")
			parsed = [float(i) for i in parsed]
			Parsed_class3.append(parsed)
		#for all 3 wine
		for index in allThreeWines:
			parsed = index.split(",")
			parsed = [float(i) for i in parsed]
			Parsed_allThreeWine.append(parsed)
			Parsed_allThreeWine2.append(parsed)

		for i in range(len(Parsed_allThreeWine2)):
			Parsed_allThreeWine2[i] = Parsed_allThreeWine2[i][1:]

		# Parsed_allThreeWine2 = Parsed_allThreeWine2[][1:]

		# for index in Parsed_allThreeWine2:
		# 	print index

		#parse each class in Iris
		for index in irisSetosaArray:
			parsed = index.split(",")
			Parsed_irisSetosaArray.append(parsed)
		for index in irisVersiColorArray:
			parsed = index.split(",")
			Parsed_irisVersiColorArray.append(parsed)
		for index in irisVirginicaArray:
			parsed = index.split(",")
			Parsed_irisVirginicaArray.append(parsed)
		#for all 3 iris
		for index in allThreeIrisArray:
			parsed = index.split(",")
			Parsed_allThree.append(parsed)
			parsed = [float(i) for i in parsed[:-1]]
			Parsed_allThree2.append(parsed)

		#deletes last newline
		Parsed_allThree = Parsed_allThree[:-1]
		Parsed_allThreeWine = Parsed_allThreeWine[:-1]

		#close the opened files
		irisFile.close()
		irisFile2.close()
		wineFile.close()
		wineFile2.close()


		#ABOVE IS ALL THE PRELIMINARIES, READING THE FILES AND SETTING IT UP. NOW IS WHERE THE PROGRAM BEINGS

		wineOrIris = input("Would you like to see the data from the Iris or Wine Dataset? Enter 1 or 2 respectively: ")
		#USER WANTS TO SEE IRIS DATASET
		if wineOrIris == 1:

			irisAttributes = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
			irisClass = ["Iris-setosa", "Iris-versicolor", "Iris-virginica", "All of the above"]


			#prompt the user to choose what they want to sort the dataset by
			for index in range(len(irisAttributes)):
				print "\t", index + 1, "- ", irisAttributes[index]
			attributeNum = input("\nWhat would you like to sort the attributes by - ")
			if attributeNum > 4 or attributeNum < 1:
				print "Invalid input. Terminating Program . . . "
				exit()
			desiredAttribute = irisAttributes[attributeNum - 1]
			print "\n\nSorted by", irisAttributes[attributeNum - 1], " . . . \n"

			#attributeNum = input("\nWhat would you like to sort the attribute by?\n\t1: Sepal Length.\n\t2: Sepal Width.\n\t3: Petal Length.\n\t4: Petal Width.\n\n\tAnswer - ")
			attributeNum -= 1

			#sort based off of the user's given input
			Parsed_irisSetosaArray.sort(key = lambda attribute: attribute[attributeNum])
			Parsed_irisVersiColorArray.sort(key = lambda attribute: attribute[attributeNum])
			Parsed_irisVirginicaArray.sort(key = lambda attribute: attribute[attributeNum])

			#sort all the whole dataset
			Parsed_allThree.sort(key = lambda attribute: attribute[attributeNum])

			#prompt the user to choose what class or iris they want
			for index in range(len(irisClass)):
				print "\t", index + 1, "- ", irisClass[index]
			numClasses = input("\nWhich dataset would you like to see - ")
			#numClasses = input("\nWhich dataset would you like to see? \n\t Iris Setosa? (1)\n\t Iris-versicolor? (2)\n\t Iris-virginica? (3) \n\t All of the above? (4) \n\n\t Answer - ")

			print irisClass[numClasses - 1], "it is!"

			if numClasses - 1 == 0:
				attributeRange = float(Parsed_irisSetosaArray[-1][attributeNum]) - float(Parsed_irisSetosaArray[0][attributeNum])
				lowerBound = float(Parsed_irisSetosaArray[0][attributeNum])
				finalDataParsed = Parsed_irisSetosaArray
			elif numClasses - 1 == 1:
				attributeRange = float(Parsed_irisVersiColorArray[-1][attributeNum]) - float(Parsed_irisVersiColorArray[0][attributeNum])
				lowerBound = float(Parsed_irisVersiColorArray[0][attributeNum])
				finalDataParsed = Parsed_irisVersiColorArray
			elif numClasses - 1 == 2:
				attributeRange = float(Parsed_irisVirginicaArray[-1][attributeNum]) - float(Parsed_irisVirginicaArray[0][attributeNum])
				lowerBound = float(Parsed_irisVirginicaArray[0][attributeNum])
				finalDataParsed = Parsed_irisVirginicaArray
			elif numClasses - 1 == 3:
				attributeRange = float(Parsed_allThree[-1][attributeNum]) - float(Parsed_allThree[0][attributeNum])
				lowerBound = float(Parsed_allThree[0][attributeNum])
				finalDataParsed = Parsed_allThree
			else:
				print "Invalid input. Terminating Program . . . "
				exit()
		elif wineOrIris == 2:

			wineAttributes = ["Alcohol", "Malic Acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color Intnesity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
			wineClass = ["Wine 1", "Wine 2", "Wine 3", "All 3"]

			for index in range(len(wineAttributes)):
				print "\t", index + 1, "- ", wineAttributes[index]
			attributeNum = input("\nWhat would you like to sort the attribute by - ")
			if attributeNum > 13 or attributeNum < 1:
				print "Invalid input. Terminating Program . . . "
				exit()
			desiredAttribute = wineAttributes[attributeNum - 1]
			print "\n\nSorted by", wineAttributes[attributeNum - 1], " . . . \n"

			#sort based off of the user's given input
			Parsed_class1.sort(key = lambda attribute: attribute[attributeNum])
			Parsed_class2.sort(key = lambda attribute: attribute[attributeNum])
			Parsed_class3.sort(key = lambda attribute: attribute[attributeNum])

			#sort all the whole dataset
			Parsed_allThreeWine.sort(key = lambda attribute: attribute[attributeNum])

			#prompt the user to choose what class or iris they want
			for index in range(len(wineClass)):
				print "\t", index + 1, "- ", wineClass[index]
			numClasses = input("\nWhich dataset would you like to see - ")

			#exit if they did not enter a correct number of classes
			if numClasses > 4 or numClasses < 1:
				print "Invalid input. Terminating Program . . . "
				exit()

			print wineClass[numClasses - 1], "it is!"

			if numClasses - 1 == 0:
				attributeRange = float(Parsed_class1[-1][attributeNum]) - float(Parsed_class1[0][attributeNum])
				lowerBound = float(Parsed_class1[0][attributeNum])
				finalDataParsed = Parsed_class1
			elif numClasses - 1 == 1:
				attributeRange = float(Parsed_class2[-1][attributeNum]) - float(Parsed_class2[0][attributeNum])
				lowerBound = float(Parsed_class2[0][attributeNum])
				finalDataParsed = Parsed_class2
			elif numClasses - 1 == 2:
				attributeRange = float(Parsed_class3[-1][attributeNum]) - float(Parsed_class3[0][attributeNum])
				lowerBound = float(Parsed_class3[0][attributeNum])
				finalDataParsed = Parsed_class3
			elif numClasses - 1 == 3:
				attributeRange = float(Parsed_allThreeWine[-1][attributeNum]) - float(Parsed_allThreeWine[0][attributeNum])
				lowerBound = float(Parsed_allThreeWine[0][attributeNum])
				finalDataParsed = Parsed_allThreeWine

		#calculate bin size by getting the range, then dividing it by # of bins
		binSize = attributeRange / binNum
		#print "BinSize = ", binSize
		#lowerbound calculated in if statements
		upperBound = lowerBound + binSize
		lowerBound2 = lowerBound
		upperBound2 = upperBound
		frequency = [0]
		index = 0;

		#calculating which datapoints go to which bin
		while index != len(finalDataParsed):
			if float(finalDataParsed[index][attributeNum]) >= lowerBound and float(finalDataParsed[index][attributeNum]) < upperBound:
					frequency[-1] = frequency[-1] + 1
					index += 1
			elif index == len(finalDataParsed) - 1 or index == 0:
				frequency[-1] += 1
				index += 1
			elif float(finalDataParsed[index][attributeNum]) >= upperBound:
				frequency.append(0)
				lowerBound += binSize
				upperBound += binSize

		print frequency
		frequencyIndex = 0

		#displaying the graphs
		while frequencyIndex <= len(frequency) - 1:
			plt.bar(lowerBound2, frequency[frequencyIndex], binSize)
			if wineOrIris == 1:
				plt.xlabel(irisClass[numClasses - 1])
				plt.ylabel(desiredAttribute)
				plt.title(irisClass[numClasses - 1] + " " + desiredAttribute + " frequency with binsize " + str(binNum))
			elif wineOrIris == 2:
				plt.xlabel(wineClass[numClasses - 1])
				plt.ylabel(wineAttributes[attributeNum - 1])
				plt.title(wineClass[numClasses - 1] + " vs " + desiredAttribute)

			lowerBound2 += binSize
			frequencyIndex += 1
		plt.show()

		plt.boxplot(frequency)
		plt.show()


		##################################################################################################################
		# UNCOMMENT AS NEEDED. PARSED_ALLTHREE2 = IRIS DATASET
		#					   PARSED_ALLTHREEWINE2 = WINE DATA SET


		#correlation(Parsed_allThree2)
		#correlation(Parsed_allThreeWine2)
		#heatMap(correlation(Parsed_allThreeWine2))
		#correlation(Parsed_allThreeWine2)
		#scatterPlot(Parsed_allThree2)
		for index in Parsed_irisSetosaArray:
			print index
		scatterPlot(Parsed_irisSetosaArray)



		# distance(x,y,p)


		##################################################################################################################
				

main()
