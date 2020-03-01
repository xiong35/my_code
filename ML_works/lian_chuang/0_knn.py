
import numpy as np


class knn:
    filename = None
    testFeat = None
    testLabels = None

    def __init__(self, filename):
        self.filename = filename
        pass

    def line2Data(self, line):
        line = line.strip('\n')
        line = line.split(',')
        curLine = []
        for feat in line[:-1]:
            curLine.append(float(feat))
        if str(line[-1]) == 'Iris-setosa':
            curLine.append(-1)
        elif str(line[-1]) == 'Iris-versicolor':
            curLine.append(0)
        elif str(line[-1]) == 'Iris-virginica':
            curLine.append(1)
        return curLine
    # 处理数据 done

    def file2matrix(self):
        with open(self.filename) as fr:
            arrayOLines = fr.readlines()[:-1]
            numOflines = len(arrayOLines)
            numOfTestData = int(numOflines*0.2)
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
            self.testFeat = np.matrix(testFeat)
            self.testLabels = np.matrix(testLabels)
            # 将"训练"数据整合成矩阵
            retMat = []
            retLabels = []
            for index in range(numOflines):
                if index in indexOfTest:
                    continue
                line = self.line2Data(arrayOLines[index])
                retMat.append(line[:-1])
                retLabels.append(line[-1])
            retMat = np.matrix(retMat)
            retLabels = np.matrix(retLabels)
            return retMat, retLabels

    def normalize(self):
        mat, labels = self.file2matrix()
        mean = mat.mean(axis=0)
        mat -= mean
        std = mat.std(axis=0)
        mat /= std

        self.testFeat -= mean
        self.testFeat /= std
        print(self.testFeat)
        return mat, labels

    # 计算测试样本与所有训练样本的距离
    # def calculateDist(self):
    #     # def classify(inX, dataSet, labels, k):
    #     data, labels = self.normalize()
    #     dataSize = data.shape[0]
    #     loss = 0

    #     dataSize = dataSet.shape[0]
    #     diffMat = np.tile(inX, (dataSize, 1)) - dataSet
    #     sqdiffMat = diffMat ** 2
    #     sqDistance = sqdiffMat.sum(axis=1)
    #     distances = sqDistance ** 0.5
    #     sortedDist = distances.argsort()
    #     classCount = {}
    #     for i in range(k):
    #         voteIlable = labels[sortedDist[i]]
    #         classCount[voteIlable] = classCount.get(voteIlable, 0) + 1
    #     sortedClassCount = sorted(classCount.items(),
    #                               key=operator.itemgetter(1), reverse=True)
    #     return sortedClassCount[0][0]

    # 对距离进行升序排序，取前k个
    def sortDist(self):
        pass

    # 计算k个样本中最多的分类
    def findMax(self):
        pass


k = knn('lian_chuang\\data\\iris.data')
k.normalize()
