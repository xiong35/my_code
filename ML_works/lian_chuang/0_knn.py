
import numpy as np
import operator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import namedtuple


class KdNode:
    def __init__(self, storePt, splitInd, lChild, rChild):
        self.storePt = storePt
        self.splitInd = splitInd
        self.lChild = lChild
        self.rChild = rChild


class KdTree:
    def __init__(self, data):
        k = len(data[0])

        def CreateNode(split, dataSet):
            if len(dataSet) == 0:
                return None
            dataSet.sort(key=lambda x: x[split])
            splitInd = len(dataSet) // 2
            midData = dataSet[splitInd]
            splitNext = (split + 1) % k

            return KdNode(midData, split,
                          CreateNode(splitNext, dataSet[:splitInd]),
                          CreateNode(splitNext, dataSet[splitInd + 1:]))

        self.root = CreateNode(0, data)


class Result:
    def __init__(self, nearestPoint, nearestDist, visitedNodes):
        self.nearestPoint = nearestPoint
        self.nearestDist = nearestDist
        self.visitedNodes = visitedNodes


class knn:
    filename = None
    testFeat = None
    testLabels = None
    data = None
    labels = None
    k = 7
    predictLabel = None
    kdTree = None

    def __init__(self, filename):
        self.filename = filename
        self.data, self.labels = self.normalize()

    def predict(self, testData=None):
        predictLabel = []
        if not testData:
            testData = self.testFeat
        for data in testData:
            print(self.findNearest(self.kdTree,data).nearestPoint)
        # self.plot(testData, predictLabel)

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
        return mat.tolist(), labels.tolist()

    def train(self):
        self.kdTree = KdTree(self.data)

    def findNearest(self, tree, point):
        k = len(point)

        def travel(kdNode, target, maxDist):
            if kdNode is None:
                result = Result(np.mat([0] * k), float("inf"), 0)
                return result

            visitedNodes = 1

            s = kdNode.splitInd
            curPt = kdNode.storePt

            if target[s] <= curPt[s]:
                nearerNode = kdNode.lChild
                furtherNode = kdNode.rChild
            else:
                nearerNode = kdNode.rChild
                furtherNode = kdNode.lChild

            temp1 = travel(nearerNode, target, maxDist)

            nearest = temp1.nearestPoint
            dist = temp1.nearestDist

            visitedNodes += temp1.visitedNodes

            if dist < maxDist:
                maxDist = dist

            tempDist = abs(curPt[s] - target[s])
            if maxDist < tempDist:
                result = Result(nearest, dist, visitedNodes)
                return result

            tempDist = np.sqrt(
                sum((p1 - p2) ** 2 for p1, p2 in zip(curPt, target)))

            if tempDist < dist:
                nearest = curPt
                dist = tempDist
                maxDist = dist

            temp2 = travel(furtherNode, target, maxDist)

            visitedNodes += temp2.visitedNodes
            if temp2.nearestDist < dist:
                nearest = temp2.nearestPoint
                dist = temp2.nearestDist
            result = Result(nearest, dist, visitedNodes)
            return result

        return travel(tree.root, point, float("inf"))

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
k.train()
k.predict()
