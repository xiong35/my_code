
import numpy as np


class knn:
    filename = None

    def __init__(self, filename):
        self.filename = filename
        pass

    # 处理数据 done
    def file2matrix(self):
        with open(self.filename) as fr:
            arrayOLines = fr.readlines()
            # numOflines = len(arrayOLines)
            # numOfFeat = len(arrayOLines[0])-1
            retMat = []
            retLabels = []
            for line in arrayOLines:
                line = line.strip('\n')
                print(line)
                if not line:
                    continue
                line = line.split(',')
                curLine = []
                for feat in line[:-1]:
                    curLine.append(float(feat))
                retMat.append(curLine)
                if str(line[-1]) == 'Iris-setosa':
                    retLabels.append(-1)
                elif str(line[-1]) == 'Iris-versicolor':
                    retLabels.append(0)
                else:
                    retLabels.append(1)
            # retLabels = np.mat(retLabels)
            retMat = np.matrix(retMat)
            return retMat, retLabels

    def normalize(self):
        mat, labels = self.file2matrix()
        print(mat)
        mean = mat.mean(axis=0)
        mat -= mean
        std = mat.std(axis=0)
        mat /= std
        return mat, labels

    # 计算测试样本与所有训练样本的距离
    def calculateDist(self):
        pass

    # 对距离进行升序排序，取前k个
    def sortDist(self):
        pass

    # 计算k个样本中最多的分类
    def findMax(self):
        pass


k = knn('lian_chuang\\data\\iris.data')
print(k.normalize())
