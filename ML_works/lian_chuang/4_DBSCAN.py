
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(777)


class DBSCAN:

    data = []
    nDim = 7
    clusters = 4

    def __init__(self):
        for _ in range(self.clusters):
            self.data.append(self.genRandData())
        print(self.data)

    def genRandData(self):
        a = np.random.uniform(-3, 3, )
        x7 = np.linspace(a-0.5, a+0.5, 100)
        thisCluster = []
        for _ in range(self.nDim):
            fn = self.genRandFunc()
            yi = []
            for x in x7:
                yi.append(fn(x)+np.random.uniform(-7, 7))
            thisCluster.append(yi)
        return thisCluster

    def genRandFunc(self):
        # ParameterList
        pL = []
        for _ in range(4):
            pL.append(np.random.uniform(-4, 4))

        def fn(i): return pL[0]*pow(i, 3)+pL[1]*pow(i, 2)+pL[2]*i+pL[3]
        return fn

    def 

d = DBSCAN()
