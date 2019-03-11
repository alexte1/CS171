import math
import numpy as np
import random
import matplotlib.pyplot as plt

def plotting(stats, errorBar, number):
	xaxis = [1,2,3,4,5,6,7,8,9,10]
	plt.plot(xaxis, stats)
	plt.xlabel('K Nearest Neighbor')
	plt.errorbar(xaxis, stats, yerr = errorBar)
	if number == 1:
		plt.ylabel('Average Accuracy Measure')
		plt.title('kNN vs. Accuracy')
	elif number == 2:
		plt.ylabel('Average Sensitiviy Measure')
		plt.title('kNN vs. Sensitiviy')
	else:
		plt.ylabel('Average Specificity Measure')
		plt.title('kNN vs. Specificity')

	plt.xlim([0, 12])
	plt.ylim([0, 1.5])
	plt.show()

def classLabel(array):
	tmp = []
	for i in range(len(array)):
		tmp.append(array[i][-1])
	return tmp

def metrics(y_pred, y_test):
	#count = 0.0
	Tp, Tn, Fp, Fn = 0.0, 0.0, 0.0, 0.0
	#calculating the %
	#assigning 2 as negative and 4 as positive
	for x,y in zip(y_pred, y_test):
		if x == y:
			if x == 2:
				Tn += 1
			else:
				Tp += 1
		else:
			if x == 2:
				Fn += 1
			else:
				Fp += 1

	accuracy = (Tp + Tn) / (Tp + Tn + Fp + Fn)
	sensitivity = Tp / (Tp + Fn)
	specificity = Tn / (Tn + Fp)

	tmp = [accuracy, sensitivity, specificity]

	return (accuracy, sensitivity, specificity)

def knn_classifier(x_test, x_train, y_train, k, p):
	distanceArray = []
	y_pred = []

	for i in range(len(x_test)):
		tmpData = x_train[:]
		tmpClass = y_train[:]
		knn = []
		while len(knn) < k:
			lowest = 9999
			index = -1
			for j in range(len(tmpData)):
				curr = distance(x_test[i], tmpData[j], p)
				if curr < lowest:
					lowest = curr
					index = j

			#stores it as a tuple
			knn.append((lowest, tmpClass[index]))
			del tmpData[index]
			del tmpClass[index]

		two = 0
		four = 0

		for l in range(len(knn)):
			if knn[l][1] == 2:
				two += 1
			else:
				four += 1
		if two >= four:
			y_pred.append(2)
		else:
			y_pred.append(4)

	return y_pred

def distance(x, y, p):
	retSum = 0
	for i,j in zip(x,y):
		retSum += abs(i-j)**p
	retSum = retSum**(1./p)
	return retSum

def eightyTwentyValues(array):

	#always round up
	eightyPercent = math.ceil(0.8 * len(array))
	twentyPerent = len(array) - eightyPercent

	return int(eightyPercent)

def main():

	file = open("breast-cancer-wisconsin", "r")

	i = 0
	tmpArray = []
	breastArray = []
	for parse in file.readlines():
		parse = parse[:-1]
		breastArray.append(parse.split(","))
		breastArray[-1] = breastArray[-1][1:]
	file.close()

	for i in range(len(breastArray)):
		for j in range(len(breastArray[i])):
			if breastArray[i][j] == "?":
				breastArray[i][j] = 0
			else:
				breastArray[i][j] = int(breastArray[i][j])

	eightyTwentyTuple = eightyTwentyValues(breastArray)

	#eightyTwentyTuple[0] holds the 80%, or training set
	trainingSet = breastArray[:eightyTwentyTuple]
	#eightyTwentyTuple[1] holds the 20%, or testing set
	testingSet = breastArray[eightyTwentyTuple:]

	print testingSet

	y_train = classLabel(trainingSet)
	y_test  = classLabel(testingSet)

	k = input("How many nearest nearest neighbors, k - ")
	p = input("Lp norm? (1 or 2) - ")
	
	y_pred = knn_classifier(testingSet, trainingSet, y_train, int(k), int(p))

	# ######################################################################################################################################################################################
	metrics(y_pred, y_test)
	# special_case_flag = True
	#splitting into 10 partitions.

	######################################################################################
	# ASSUMPTION - SINCE 699 / 10 = 69.9, I just made 9 partitions have 70 and 1 have 69 #
	######################################################################################

	foldSize = int(round(len(breastArray) / 10.0))

	for p in range(2):

		AccuracyPerFold = []
		SensitivityPerFold = []
		SpecificityPerFold = []
		stdAcc = []
		stdSen = []
		stdSpec = []

		for k in range(10):

			tmpAccuracy = []
			tmpSensitivity = []
			tmpSpecificity = []

			for i in range(10):
				testSet = []
				ten_Fold = breastArray[:]	#copies it over so we can manipulate ten_fold
				if i == 9:
					for j in breastArray[i*foldSize:]:
						testSet.append(j)
					del ten_Fold[i*foldSize:]
				else:
					for j in breastArray[i*foldSize:(i+1)*foldSize]:
						testSet.append(j)
					del ten_Fold[i*foldSize:(i+1)*foldSize]

				y_train = classLabel(ten_Fold)
				y_test = classLabel(testSet)
				y_pred = knn_classifier(testSet, ten_Fold, y_train, int(k + 1), int(p + 1))

				stats = metrics(y_pred, y_test)
				tmpAccuracy.append(stats[0])	#takes the first accuracy
				tmpSensitivity.append(stats[1])
				tmpSpecificity.append(stats[2])
			AccuracyPerFold.append(np.mean(tmpAccuracy))
			SensitivityPerFold.append(np.mean(tmpSensitivity))
			SpecificityPerFold.append(np.mean(tmpSpecificity))
			stdAcc.append(np.std(tmpAccuracy))
			stdSen.append(np.std(tmpSensitivity))
			stdSpec.append(np.std(tmpSpecificity))
		plotting(AccuracyPerFold,stdAcc, 0)
		plotting(SensitivityPerFold, stdSen, 1)
		plotting(SpecificityPerFold, stdSpec, 2)

main()