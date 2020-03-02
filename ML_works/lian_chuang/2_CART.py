
import numpy as np
import pandas as pd


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


def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = np.dot(X, ws)
    return np.var(yHat-Y)


def linearSolve(dataSet):
    m, n = np.shape(dataSet)
    X = np.mat(np.ones((m, n)))
    Y = np.mat(np.ones((m, 1)))
    X[:, 1:n] = dataSet[:, 1:n]
    Y = dataSet[:, 0]
    xTx = np.mat(np.dot(X.T, X))
    if np.linalg.det(xTx) == 0:
        raise NameError('This matrix is singular, cannot do inverse, \
            try to increase the second value of ops')
    ws = np.dot(xTx.I, np.dot(X.T, Y))
    return ws, X, Y


def regTreeEval(model, inDat):
    return float(model)


def modelTreeEval(model, inDat):
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1, n+1)))
    X[:, 1: n+1] = inDat
    return float(np.dot(X[:, 1:], model))


class CART:
    Tree = None
    dataSet = None
    testData = None

    def __init__(self, filename):
        df = pd.read_table(filename, header=0, sep=" ", dtype=str)
        df.drop(columns=['PassengerId', 'Name',
                         'Ticket', 'Cabin'], axis=1, inplace=True)
        df.replace(to_replace=np.nan, value='40', regex=True, inplace=True)
        df.replace(to_replace='C', value='0', regex=True, inplace=True)
        df.replace(to_replace='S', value='1', regex=True, inplace=True)
        df.replace(to_replace='Q', value='-1', regex=True, inplace=True)
        df.replace(to_replace='female', value='0', regex=True, inplace=True)
        df.replace(to_replace='male', value='1', regex=True, inplace=True)
        self.dataSet = np.mat(df).astype('float32')
        self.testData = self.dataSet[:150]
        self.dataSet = self.dataSet[150:]

    def binSplitDataSet(self, dataSet, feature, value):
        mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
        mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
        return mat0, mat1

    def createTree(self, dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
        dataSet = np.mat(dataSet)
        feat, val = self.chooseBestSplit(dataSet, leafType, errType, ops)
        if feat == None:
            return val
        retTree = TreeNode()
        retTree.spInd = feat
        retTree.spVal = val
        lSet, rSet = self.binSplitDataSet(dataSet, feat, val)
        retTree.lChild = self.createTree(lSet, leafType, errType, ops)
        retTree.rChild = self.createTree(rSet, leafType, errType, ops)
        return retTree

    def chooseBestSplit(self, dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
        # minimum step of descent
        tolS = ops[0]
        # minimum num of samples to split
        tolN = ops[1]
        if len(set(dataSet[:, 0].T.tolist()[0])) == 1:
            return None, leafType(dataSet)
        m, n = np.shape(dataSet)
        S = errType(dataSet)
        bestS = np.inf
        bestIndex = 0
        bestValue = 0
        for featIndex in range(1, n):
            for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
                mat0, mat1 = self.binSplitDataSet(dataSet, featIndex, splitVal)
                if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                    continue
                newS = errType(mat0) + errType(mat1)
                if newS < bestS:
                    bestIndex = featIndex
                    bestValue = splitVal
                    bestS = newS
        if (S - bestS) < tolS:
            return None, leafType(dataSet)
        mat0, mat1 = self.binSplitDataSet(dataSet, bestIndex, bestValue)
        if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
            return None, leafType(dataSet)
        return bestIndex, bestValue

    def createFore(self, tree, modelEval=regTreeEval):
        m = len(self.testData)
        yHat = np.mat(np.zeros((m, 1)))
        for i in range(m):
            yHat[i, 0] = self.treeForecast(tree, self.testData[i], modelEval)
        return yHat

    def treeForecast(self, tree, inData, modelEval):
        if not isTree(tree):
            return modelEval(tree, inData)
        # inData is a 1*m matrix
        if inData.tolist()[0][tree.spInd] > tree.spVal:
            if isTree(tree.lChild):
                return self.treeForecast(tree.lChild, inData, modelEval)
            else:
                return modelEval(tree.lChild, inData)
        else:
            if isTree(tree.rChild):
                return self.treeForecast(tree.rChild, inData, modelEval)
            else:
                return modelEval(tree.rChild, inData)

    def predict(self):
        tree = self.createTree(self.dataSet)
        predVec = self.createFore(tree)
        pred = []
        acc = 0
        for i in range(len(self.testData)):
            if predVec[i][0] > 0.5:
                pred.append(1)
            else:
                pred.append(0)
            if pred[i] == self.testData[i,0]:
                acc += 1
        acc /= len(self.testData)
        print('acc: ',acc)


def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0


def prune(tree, testData):
    if np.shape(testData)[0] == 0:
        return getMean(tree)
    if isTree(tree['left']) or isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData,
                                     tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if (not isTree(tree['left'])) and (not isTree(tree['right'])):
        lSet, rSet = binSplitDataSet(testData,
                                     tree['spInd'], tree['spVal'])
        errorNoMerge = sum(np.power(lSet[:, 0]-tree['left'], 2)) +\
            sum(np.power(rSet[:, 0]-tree['right'], 2))
        treeMean = (tree['left']+tree['right'])/2
        errorMerge = sum(np.power(testData[:, 0]-treeMean, 2))
        if errorNoMerge > errorMerge:
            print("merge")
            return treeMean
        else:
            return tree
    else:
        return tree


c = CART(R'lian_chuang\data\titanic.txt')
c.predict()


# myTree = createTree(myData, modelLeaf, modelErr, ops=(0.1, 2))
# # print(myTree)

# model = modelTreeEval  # /regTreeEval
# # remenber to change the leaf/err too

# yHat = createFore(myTree, testData, model)
# # print(yHat)


# # prune(myTree, testData)
