
import numpy as np
import pandas as pd


class NaiveBayes:
    dataSet = []
    labels = []
    posibilityMatOf0 = []
    posibilityMatOf1 = []
    NumOf1 = 0
    NumOf0 = 0

    def __init__(self, filename):
        df = pd.read_table(filename, header=0, sep=" ",
                           keep_default_na=False, dtype=str)
        df.drop(columns=['PassengerId', 'Name',
                         'Ticket', 'Cabin'], axis=1, inplace=True)
        self.labels = df.values[:, 0]
        self.dataSet = df.values[:, 1:]

    def holdOutSplit(self):
        numOfDate = len(self.dataSet)
        numOfTestData = int(numOfDate*0.2)
        testLabel = self.labels[:numOfTestData]
        testData = self.dataSet[:numOfTestData]
        trainLabel = self.labels[numOfTestData:]
        trainData = self.dataSet[numOfTestData:]
        return (trainData, trainLabel), (testData, testLabel)

    def crossVal(self, i):
        numPerFold = len(self.dataSet)//5
        testData = self.dataSet[i*numPerFold:(i+1)*numPerFold]
        trainData = np.concatenate((self.dataSet[:i*numPerFold],
                                    self.dataSet[(i+1)*numPerFold:]))
        testLabel = self.labels[i*numPerFold:(i+1)*numPerFold]
        trainLabel = np.concatenate((self.labels[:i*numPerFold],
                                     self.labels[(i+1)*numPerFold:]))
        return (trainData, trainLabel), (testData, testLabel)

    def boostStrap(self):
        randSet = set()
        trainData = []
        trainLabel = []
        for _ in range(len(self.dataSet)):
            rand = np.random.randint(0, len(self.dataSet))
            randSet.add(rand)
            trainData.append(self.dataSet.tolist()[rand])
            trainLabel.append(self.labels[rand])
        testData = []
        testLabel = []
        for i in range(len(self.dataSet)):
            if i in randSet:
                continue
            testData.append(self.dataSet.tolist()[i])
            testLabel.append(self.labels[i])
        trainLabel = np.array(trainLabel)
        trainData = np.array(trainData)
        testLabel = np.array(testLabel)
        testData = np.array(testData)
        return (trainData, trainLabel), (testData, testLabel)

    def strPosibility(self, oneColOfData, trainLabel):
        strSet = set()
        oneColOfData = oneColOfData.tolist()
        for string in oneColOfData:
            if string not in strSet:
                strSet.add(string)
        strNum = len(strSet)
        p0Num = np.zeros(strNum)
        p1Num = np.zeros(strNum)
        for index, string in enumerate(oneColOfData):
            if trainLabel[index] == '1':
                p1Num[oneColOfData.index(string)] += 1
            else:
                p0Num[oneColOfData.index(string)] += 1
        p0Pos = np.log(p0Num/self.NumOf0+1e-5)
        p1Pos = np.log(p1Num/self.NumOf1+1e-5)

        self.posibilityMatOf0.append(p0Pos)
        self.posibilityMatOf1.append(p1Pos)

    def numPosibility(self,oneColOfData, trainLabel):
        minNum =  oneColOfData.min()
        maxNum = oneColOfData.max()
        step = (maxNum - minNum)/10
        p0Num = np.ones(10)
        p1Num = np.ones(10)
        for index, num in enumerate(oneColOfData):
            if trainLabel[index] == '1':



    def train(self):
        # code for training your naive bayes algorithm
        (trainData, trainLabel), (testData, testLabel) = self.holdOutSplit()
        for i in range(len(trainLabel)):
            if trainLabel[i] == '1':
                self.NumOf1 += 1
            else:
                self.NumOf0 += 1
        self.strPosibility(trainData[:, 3], trainLabel)

    def predict(self):
        # code for predicting
        pass


n = NaiveBayes(R'lian_chuang\data\titanic.txt')
(a, b), (c, d) = n.boostStrap()
n.train()
