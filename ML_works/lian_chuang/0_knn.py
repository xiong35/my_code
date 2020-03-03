
import numpy as np
import operator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class knn:
    filename = None
    testFeat = None
    testLabels = None
    data = None
    labels = None
    k = 7
    predictLabel = None

    def __init__(self, filename):
        self.filename = filename
        self.data, self.labels = self.normalize()

    def predict(self, testData=None):
        predictLabel = []
        if not testData:
            testData = self.testFeat
        for data in testData:
            predictLabel.append(self.findMax(data))
        self.plot(testData,predictLabel)

    def line2Data(self, line):
        line = line.strip('\n')
        line = line.split(',')
        curLine = []
        for feat in line[:-1]:
            curLine.append(float(feat))
        if str(line[-1]) == 'Iris-setosa':
            curLine.append('r')
        elif str(line[-1]) == 'Iris-versicolor':
            curLine.append('g')
        elif str(line[-1]) == 'Iris-virginica':
            curLine.append('b')
        return curLine
    # 处理数据 done

    def file2matrix(self):
        with open(self.filename) as fr:
            arrayOLines = fr.readlines()[:-1]
            numOflines = len(arrayOLines)
            numOfTestData = int(numOflines*0.15)
            np.random.seed(7)
            # 随机抽测试数据
            indexOfTest = np.random.randint(0, numOflines, numOfTestData)
            testFeat = []
            testLabels = []
            # 将测试数据整合成矩阵
            for index in indexOfTest:
                line = self.line2Data(arrayOLines[index])
                testFeat.append(line[:-1])
                testLabels.append(line[-1])
            self.testFeat = np.array(testFeat)
            self.testLabels = np.array(testLabels)
            # 将"训练"数据整合成矩阵
            retMat = []
            retLabels = []
            for index in range(numOflines):
                if index in indexOfTest:
                    continue
                line = self.line2Data(arrayOLines[index])
                retMat.append(line[:-1])
                retLabels.append(line[-1])
            retMat = np.array(retMat)
            retLabels = np.array(retLabels)
            return retMat, retLabels

    def normalize(self):
        mat, labels = self.file2matrix()
        mean = mat.mean(axis=0)
        mat -= mean
        std = mat.std(axis=0)
        mat /= std

        self.testFeat -= mean
        self.testFeat /= std
        return mat, labels

    # 计算测试样本与所有训练样本的距离
    def findMax(self, testData):
        diffMat = self.data-testData
        sqDiffMat = diffMat ** 2
        sqDistance = sqDiffMat.sum(axis=1)
        distances = sqDistance ** 0.5
        sortedDist = distances.argsort()
        classCount = {}
        for i in range(self.k):
            voteIlable = self.labels[sortedDist[i]]
            classCount[voteIlable] = classCount.get(voteIlable, 0) + 1
        sortedClassCount = sorted(classCount.items(),
                                  key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    def plot(self, testData, predictLabel):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        featX = self.data[:, 0].tolist()
        featY = self.data[:, 1].tolist()
        featZ = self.data[:, 2].tolist()
        originLabels = list(self.labels)
        ax.scatter3D(featX, featY, featZ, c=originLabels, alpha=0.25)
        testFeatX = testData[:, 0].tolist()
        testFeatY = testData[:, 1].tolist()
        testFeatZ = testData[:, 2].tolist()
        ax.scatter3D(testFeatX, testFeatY, testFeatZ,
                     c=list(predictLabel), marker='+', alpha=1)
        plt.show()


k = knn('lian_chuang\\data\\iris.data')
k.predict()