
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(7)

class opStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = len(dataMatIn)
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))


def loadDataSet(filename):
    df = pd.read_table(filename, header=0, sep=" ", dtype=str)
    # df.drop(columns=['PassengerId', 'Name',
    #                  'Ticket', 'Cabin'], axis=1, inplace=True)
    # df.replace(to_replace='C', value='0', regex=True, inplace=True)
    # df.replace(to_replace='S', value='1', regex=True, inplace=True)
    # df.replace(to_replace='Q', value='-1', regex=True, inplace=True)
    df = df[['Survived', 'Sex', 'Age', 'Fare']]
    df.replace(to_replace=np.nan, value='40', regex=True, inplace=True)
    df.replace(to_replace='female', value='0', regex=True, inplace=True)
    df.replace(to_replace='male', value='1', regex=True, inplace=True)
    print(df)
    dataSet = np.mat(df).astype('float32')
    classLabels = dataSet[:, 0]
    for i in range(len(classLabels)):
        if classLabels[i] == 0:
            classLabels[i] = -1
    dataSet = dataSet[:, 1:]
    dataSet = normalize(dataSet)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    featX = dataSet[:, 0].T.tolist()[0]
    featY = dataSet[:, 1].T.tolist()[0]
    featZ = dataSet[:, 2].T.tolist()[0]
    labels = classLabels.T.tolist()[0]
    ax.scatter3D(featX, featY, featZ, c=labels, alpha=0.2,cmap='viridis')
    plt.show()
    return dataSet[:200], classLabels[:200]


def normalize(dataSet):
    mean = dataSet.mean(axis=0)
    dataSet -= mean
    std = dataSet.std(axis=0)
    dataSet /= std
    return dataSet


def calEk(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T
                * (oS.X*oS.X[k, :].T))+oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):
    maxK, maxDeltaE, Ej = -1, 0, 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calEk(oS, k)
            deltaE = abs(Ei-Ek)
            if deltaE > maxDeltaE:
                maxK, maxDeltaE, Ej = k, deltaE, Ek
        return maxK, Ej
    else:
        j = i
        while j == i:
            j = np.random.randint(0, oS.m)
        Ej = calEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    Ek = calEk(oS, k)
    oS.eCache[k] = [1, Ek]


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def innerL(i, oS):
    Ei = calEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C))or\
            ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j]-oS.alphas[i])
            H = min(oS.C, oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L = max(0, oS.alphas[j]+oS.alphas[i]-oS.C)
            H = min(oS.C, oS.alphas[j]+oS.alphas[i])
        if L == H:
            return 0
        eta = 2*oS.X[i, :]*oS.X[j, :].T - oS.X[i, :] * \
            oS.X[i, :].T-oS.X[j, :]*oS.X[j, :].T
        if eta >= 0:
            print('eta>=0')
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if abs(oS.alphas[j]-alphaJold) < 1e-5:
            print('j not moving enough')
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i] *\
            (alphaJold-oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - \
            oS.labelMat[i]*(oS.alphas[i]-alphaIold) * oS.X[i, :]*oS.X[i, :].T -\
            oS.labelMat[j]*(oS.alphas[j]-alphaJold) * oS.X[i, :]*oS.X[j, :].T
        b2 = oS.b - Ej - \
            oS.labelMat[i]*(oS.alphas[i]-alphaIold) * oS.X[i, :]*oS.X[j, :].T -\
            oS.labelMat[j]*(oS.alphas[j]-alphaJold) * oS.X[j, :]*oS.X[j, :].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j])and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1+b2)/2
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = opStruct(dataMatIn, classLabels, C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0)or entireSet):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
            print('fullset, iter: {} i: {}, pair: {}'.format(
                iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0)*(oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print('non-bound, iter: {}, i: {}, pair: {}'.format(iter,
                                                                    i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
    return oS.b, oS.alphas


def calcWs(alphas, data, classLabels):
    X = data
    labelMat = classLabels
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i]*labelMat[i], X[i, :].T)
    return w


def test(ws, b, data, labels):
    acc = 0
    ws = np.mat(ws)
    for i in range(100):
        if data[i]*ws+b > 0:
            pred = 1
        else:
            pred = -1
        if pred == labels[i]:
            acc += 1
    print(acc)


def plot():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    featX = data[:, 0].tolist()
    featY = data[:, 1].tolist()
    featZ = data[:, 2].tolist()
    originLabels = list(labels)
    ax.scatter3D(featX, featY, featZ, c=originLabels, alpha=0.2)
    testFeatX = testFeat[:, 0].tolist()
    testFeatY = testFeat[:, 1].tolist()
    testFeatZ = testFeat[:, 2].tolist()
    predictLabel = list(predictLabel)
    ax.scatter3D(testFeatX, testFeatY, testFeatZ,
                 c=predictLabel, marker='+', alpha=1)
    plt.show()


filename = R'lian_chuang\data\titanic.txt'
data, classLabels = loadDataSet(filename)
b, alphas = smoP(data, classLabels, 0.6, 0.001, 40)
w = calcWs(alphas, data, classLabels)
test(w, b, data, classLabels)
