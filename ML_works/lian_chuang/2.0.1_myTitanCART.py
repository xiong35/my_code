
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TreeNode:
    lChild = None
    rChild = None
    spVal = None
    spInd = None


def isTree(tree):
    if tree.__class__ == TreeNode:
        return True
    return False


def regLeaf(dataSet):
    return np.mean(dataSet[:, 0])


def regErr(dataSet):
    # total error, not MSE
    return np.var(dataSet[:, 0])*len(dataSet)


def regTreeEval(model, inDat):
    return float(model)


class CART:
    Tree = None
    dataSet = None
    testData = None

    def __init__(self, filename):
        df = pd.read_csv(filename, header=0, sep=",", dtype=float)
        self.dataSet = np.mat(df)
        self.testData = self.dataSet[:150]
        self.dataSet = self.dataSet[150:]

    def binSplitDataSet(self, dataSet, feature, value):
        mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
        mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
        return mat0, mat1

    def createTree(self, dataSet, ops=(1.5, 5)):
        dataSet = np.mat(dataSet)
        feat, val = self.chooseBestSplit(dataSet, ops)
        if feat == None:
            return val
        retTree = TreeNode()
        retTree.spInd = feat
        retTree.spVal = val
        lSet, rSet = self.binSplitDataSet(dataSet, feat, val)
        retTree.lChild = self.createTree(lSet, ops)
        retTree.rChild = self.createTree(rSet, ops)
        return retTree

    def chooseBestSplit(self, dataSet, ops=(1.5, 5)):
        # minimum step of descent
        tolS = ops[0]
        # minimum num of samples to split
        tolN = ops[1]
        if len(set(dataSet[:, 0].T.tolist()[0])) == 1:
            return None, regLeaf(dataSet)
        _, n = np.shape(dataSet)
        S = regErr(dataSet)
        bestS = np.inf
        bestIndex = 0
        bestValue = 0
        for featIndex in range(1, n):
            for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
                mat0, mat1 = self.binSplitDataSet(dataSet, featIndex, splitVal)
                if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                    continue
                newS = regErr(mat0) + regErr(mat1)
                if newS < bestS:
                    bestIndex = featIndex
                    bestValue = splitVal
                    bestS = newS
        if (S - bestS) < tolS:
            return None, regLeaf(dataSet)
        mat0, mat1 = self.binSplitDataSet(dataSet, bestIndex, bestValue)
        if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
            return None, regLeaf(dataSet)
        return bestIndex, bestValue

    def createFore(self, tree):
        m = len(self.testData)
        yHat = np.mat(np.zeros((m, 1)))
        for i in range(m):
            yHat[i, 0] = self.treeForecast(tree, self.testData[i])
        return yHat

    def treeForecast(self, tree, inData):
        if not isTree(tree):
            return regTreeEval(tree, inData)
        # inData is a 1*m matrix
        if inData.tolist()[0][tree.spInd] > tree.spVal:
            if isTree(tree.lChild):
                return self.treeForecast(tree.lChild, inData)
            else:
                return regTreeEval(tree.lChild, inData)
        else:
            if isTree(tree.rChild):
                return self.treeForecast(tree.rChild, inData)
            else:
                return regTreeEval(tree.rChild, inData)

    def predict(self):
        tree = self.createTree(self.dataSet)
        tree = self.prune(tree, self.testData)
        self.Tree = tree
        predVec = self.createFore(tree)
        pred = []
        acc = 0
        for i in range(len(self.testData)):
            if predVec[i][0] > 0.24:
                pred.append(1)
            else:
                pred.append(-1)
            if pred[i]*self.testData[i, 0] > 0:
                acc += 1
        acc /= len(self.testData)
        print('acc: ', acc)

    def prune(self, tree, testData):
        if np.shape(testData)[0] == 0:
            return self.getMean(tree)
        if isTree(tree.lChild) or isTree(tree.rChild):
            lSet, rSet = self.binSplitDataSet(testData,
                                              tree.spInd, tree.spVal)
        if isTree(tree.lChild):
            tree.lChild = self.prune(tree.lChild, lSet)
        if isTree(tree.rChild):
            tree.rChild = self.prune(tree.rChild, rSet)
        if (not isTree(tree.lChild)) and (not isTree(tree.rChild)):
            lSet, rSet = self.binSplitDataSet(testData,
                                              tree.spInd, tree.spVal)
            errorNoMerge = sum(np.power(lSet[:, 0]-tree.lChild, 2)) +\
                sum(np.power(rSet[:, 0]-tree.rChild, 2))
            treeMean = (tree.lChild+tree.rChild)/2
            errorMerge = sum(np.power(testData[:, 0]-treeMean, 2))
            if errorNoMerge > errorMerge:
                print("merge")
                return treeMean
            else:
                return tree
        else:
            return tree

    def getMean(self, tree):
        if isTree(tree.rChild):
            tree.rChild = self.getMean(tree.rChild)
        if isTree(tree.lChild):
            tree.lChild = self.getMean(tree.lChild)
        return (tree.lChild+tree.rChild)/2.0


c = CART(R'lian_chuang\data\myTitanic.csv')
c.predict()


decisionNode = dict(boxstyle='round', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


def getNumLeafs(myTree):
    numLeafs = 0
    if not isTree(myTree):
        return 1
    numLeafs += getNumLeafs(myTree.lChild)+getNumLeafs(myTree.rChild)
    return numLeafs


def getTreeDepth(myTree):
    if not isTree(myTree):
        return 1
    return max(getTreeDepth(myTree.lChild), getTreeDepth(myTree.rChild))+1


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    plt.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                 xytext=centerPt, textcoords='axes fraction',
                 va='center', ha='center', size=6, bbox=nodeType, arrowprops=arrow_args)


def plotMidText(centerPt, parentPt, txtString):
    xMid = (parentPt[0] - centerPt[0]) / 2.0 + centerPt[0]
    yMid = (parentPt[1] - centerPt[1]) / 2.0 + centerPt[1]
    plt.text(xMid, yMid, txtString, size=6,)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    # depth = getTreeDepth(myTree)
    firstStr = 'id: ' + str(myTree.spInd)
    centerPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW,
                plotTree.yOff)
    plotMidText(centerPt, parentPt, nodeTxt)
    plotNode(firstStr, centerPt, parentPt, decisionNode)
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD

    if isTree(myTree.lChild):
        plotTree(myTree.lChild, centerPt, '<%.2f' % myTree.spVal)
    else:
        plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
        plotNode('0' if myTree.lChild < 0.24 else '1', (plotTree.xOff,
                                                        plotTree.yOff), centerPt, leafNode)
        plotMidText((plotTree.xOff, plotTree.yOff),
                    centerPt, '<%.2f' % myTree.spVal)

    if isTree(myTree.rChild):
        plotTree(myTree.rChild, centerPt, '>%.2f' % myTree.spVal)
    else:
        plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
        plotNode('0' if myTree.rChild < 0.24 else '1', (plotTree.xOff,
                                                        plotTree.yOff), centerPt, leafNode)
        plotMidText((plotTree.xOff, plotTree.yOff),
                    centerPt, '>%.2f' % myTree.spVal)
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    plt.axis('off')
    plt.subplots(111, frameon=False)
    plt.subplots()
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.axis('off')
    plt.show()


createPlot(c.Tree)
