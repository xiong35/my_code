
import numpy as np


def loadDataSet(fileName):
    detaMatrix = []
    labelMatrix = []
    with open(fileName) as fr:
        for line in fr.readlines()[:250]:
            lineArray = line.strip().split(',')
            detaMatrix.append([1.0, float(lineArray[9]),
                               float(lineArray[22])])
            if lineArray[-1] == 'b':
                labelMatrix.append(1)
            else:
                labelMatrix.append(0)
    return detaMatrix, labelMatrix


dataMatrix, labelMatrix = loadDataSet(
    '/home/ylxiong/Documents/ionosphere.data')


def sigmoid(inX):
    return 1.0/(1.0+np.exp(-1.0*inX))


def gradientAscent(dataMatrix, classMatrix):
    dataMatrix = np.mat(dataMatrix)
    labelMatrix = np.mat(classMatrix).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        # dataMatrix: 250 * 3
        # weight    : 3 * 1
        h = sigmoid(np.dot(dataMatrix, weights))
        error = (labelMatrix - h)
        weights = weights + alpha*np.dot(dataMatrix.transpose(), error)
    return weights


weights = gradientAscent(dataMatrix, labelMatrix)


def plotBestFit(wei):
    import matplotlib.pyplot as plt
    weights = wei.getA()
    dataMatrix, labelMatrix = \
        loadDataSet('/home/ylxiong/Documents/ionosphere.data')
    n = len(dataMatrix)
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMatrix[i]) == 1:
            xcord1.append(dataMatrix[i][1])
            ycord1.append(dataMatrix[i][2])
        else:
            xcord2.append(dataMatrix[i][1])
            ycord2.append(dataMatrix[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='blue')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


plotBestFit(weights)
