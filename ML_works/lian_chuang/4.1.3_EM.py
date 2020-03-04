
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def createOneDataSet(n=200, mean=[0, 0], cov=[[1, 0], [0, 1]]):
    return np.random.multivariate_normal(mean, cov, n)


# createDataSet
def createDataSet(k):
    mean = [np.random.uniform(-10, 10), np.random.uniform(-10, 10)]
    cov = np.random.randn(2, 2)
    n = np.random.randint(200, 500)
    dataSet = createOneDataSet(n, mean, cov)
    for _ in range(k-1):
        mean = [np.random.uniform(-10, 10), np.random.uniform(-10, 10)]
        cov = np.random.randn(2, 2)
        n = np.random.randint(50, 500)
        dataSet = np.concatenate(
            (dataSet, createOneDataSet(n, mean, cov)), axis=0)
    return dataSet


def randSample(dataSet):
    numOfData = len(dataSet)
    trainData, testData = [], []
    testInd = np.random.randint(0, numOfData, int(numOfData*0.2))
    for i in range(numOfData):
        if i in testInd:
            testData.append(dataSet[i])
        else:
            trainData.append(dataSet[i])
    trainData = np.mat(trainData)
    testData = np.mat(testData)
    return trainData, testData


n_cluster = 10
dataSet = createDataSet(n_cluster)
X_train, X_test = randSample(dataSet)
n_samples, n_feature = X_train.shape

# 随机初始化均值，维度为(n_cluster, n_feature)
# 生成范围/2是为了限制初始均值的生成范围
mu = np.random.randint(X_train.min()/2, X_train.max() /
                       2, size=(n_cluster, n_feature))

# 一个协方差矩阵的维度为(n_feature,n_feature)
# 多个分布的协方差矩阵维度为(n_cluster,n_feature,n_feature)
cov = np.zeros((n_cluster, n_feature, n_feature))
for dim in range(len(cov)):
    np.fill_diagonal(cov[dim], 1)

# 初始均匀的类分布概率
pi = np.ones(n_cluster)/n_cluster

# 概率矩阵
P_mat = np.zeros((n_samples, n_cluster))


max_iter = 30
for i in range(max_iter):
    # 对每一组参数进行计算
    for k in range(n_cluster):
        # 实时生成高斯分布，免去了存储
        g = multivariate_normal(mean=mu[k], cov=cov[k])

    # E-step，计算概率
        # 计算X在各分布下出现的频率
        P_mat[:, k] = pi[k]*g.pdf(X_train)

    # 计算各样本出现的总频率
    totol_N = P_mat.sum(axis=1)
    # 如果某一样本在各类中的出现频率和为0，则使用K来代替，相当于分配等概率
    totol_N[totol_N == 0] = n_cluster
    P_mat /= totol_N.reshape(-1, 1)

    # M-step，更新参数
    for k in range(n_cluster):
        N_k = np.sum(P_mat[:, k], axis=0)    # 类出现的频率
        mu[k] = (1/N_k)*np.sum(np.multiply(X_train, P_mat[:, k].reshape(-1, 1)),
                               axis=0)    # 该类的新均值
        cov[k] = ((1/N_k)*np.dot(np.multiply(P_mat[:, k].reshape(-1, 1), (X_train-mu[k])).T,
                                 (X_train-mu[k])))
        pi[k] = N_k/n_samples


# 迭代更新好参数之后，开始预测未知数据的类
pred_mat = np.zeros((X_test.shape[0], n_cluster))
for k in range(n_cluster):
    g = multivariate_normal(mean=mu[k], cov=cov[k])
    pred_mat[:, k] = pi[k]*g.pdf(X_test)

totol_N = pred_mat.sum(axis=1)
totol_N[totol_N == 0] = n_cluster
pred_mat /= totol_N.reshape(-1, 1)
Y_pred = np.argmax(pred_mat, axis=1)
plt.clf()
plt.scatter(X_test[:, 0].T.tolist()[0],
            X_test[:, 1].T.tolist()[0], c=Y_pred, alpha=0.5)
plt.show()
