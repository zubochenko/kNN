"""
kNN(train, test, k, distance)

train - train data
test - classifying object without target mark
k - number of k's neighbors
distance - 'euclid' or 'manhattan' distance
"""

import math
import pandas as pd
import numpy as np
import seaborn as sns

iris = sns.load_dataset("iris")

def kNN(train, test, k, distance): 
    
    def euclid(train, test, length):
        dist = 0
        for x in range(length-1):
            dist += (math.sqrt((train[x] - test[x])**2 + (train[x+1] - test[x+1])**2))
        return dist

    def manhattan(train, test, length):
        dist = 0
        for x in range(length-1):
            dist += sum(abs(train[x]-test[x]) for train[x],test[x] in zip(train,test))
        return dist

    def getter(train, test, k):
        distances = []
        length = len(test)
        for x in range(len(train)):
            dist = euclid(test, train[x], length)
            distances.append((train[x], dist))
        distances.sort(key=lambda j: j[1])
        NN = []
        for x in range(k):
            NN.append(distances[x][0])
        return NN
    def classFinder(nn):
        diction = {}
        for i in range(len(nn)):
            k = nn[i][-1]
            if k in diction:
                diction[k] += 1
            else:
                diction[k] = 1
        diction = sorted(diction.items(), key=lambda j: j[1], reverse=True)
        return diction[0][0]
    
    return classFinder(getter(train, test, k))
kNN (iris.values[:100], iris.values[52][:4], 1, 'euclid')

prediction = []
def getAccuracy(test, prediction, k, distance):
    
    def answers(test, prediction, k):
        for x in range(len(test)):
            res = kNN(iris.values, iris.values[x][:4], k, distance)
            prediction.append(res)
        return prediction
    answers(iris.values, prediction, k)

    def getpercent(test, predictions):
        correctAnswer = 0
        for x in range(len(test)):
            if test[x][-1] == predictions[x]:
                correctAnswer += 1
        return (correctAnswer/float(len(test))) * 100.0
    
    return getpercent(test, answers(test, prediction, k))

print (getAccuracy(iris.values, prediction, 3, 'manhattan'))
