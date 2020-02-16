
# 朴素贝叶斯实现文本分类

## 前期准备

导入numpy

    import numpy as np

自己写几句话并打上标签

    # preprocessing word index
    def loadDataSet():
        postingList = [
            ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
            ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
            ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
            ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
            ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
            ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'],
        ]
        classVec = [0, 1, 0, 1, 0, 1]
        return postingList, classVec

## 用向量表示句子

将所有出现过的词列成表

    # creat a list that contains all the words
    def createVocabList(dataSet):
        vocabSet = set([])
        for document in dataSet:
            vocabSet = vocabSet | set(document)
        return list(vocabSet)
    # for each comment, create a vector
    # vocabList rf the set of all the elements
    # inputSet  rf the set to be converted to vector

方法1：one-hot编码

    def bagOfWords2Vec(vocabList, inputSet):
        returnVec = [0]*len(vocabList)
        for word in inputSet:
            if word in vocabList:
                returnVec[vocabList.index(word)] = 1
        return returnVec

方法2：bag of words

    def bagOfWords2Vec(vocabList, inputSet):
        returnVec = [0]*len(vocabList)
        for word in inputSet:
            if word in vocabList:
                returnVec[vocabList.index(word)] += 1
        return returnVec

以上两种方法取其一即可

## 朴素贝叶斯

p(c|w) = \frac{p(w|c)\cdot p(c)}{p(w)}  

其中：c代表category，w代表words  

### 具体实现

    # p1 rf p(negative)
    def trainNB0(trainMatrix, trainCategory):
        numTrainDocs = len(trainMatrix)
        numWords = len(trainMatrix[0])
        # sum of this vector [trainCategory]
        # is the num of negative cases,
        # because negative are present in [1]
        # and 1-pNegative is pPositive
        pNegative = sum(trainCategory)/float(numTrainDocs)
        # in case of [zero cases], set each word to 1 at first
        p0Num = np.ones(numWords)
        p1Num = np.ones(numWords)
        p0Denom = 2.0
        p1Denom = 2.0
        for i in range(numTrainDocs):
            # if the class is negative
            if trainCategory[i] == 1:
                # for each word in the comment
                # mark it in the vector
                p1Num += trainMatrix[i]
                # the num of all negative words++
                p1Denom += sum(trainMatrix[i])
            else:
                p0Num += trainMatrix[i]
                p0Denom += sum(trainMatrix[i])
        # elementwise deviation
        # p1Vect = p1Num/p1Denom
        # p0Vect = p0Num/p0Denom
        # too many small num doing multiplication leads to overflow
        # change to log, do no harm to the maximum
        p1Vect = np.log(p1Num/p1Denom)
        p0Vect = np.log(p0Num/p0Denom)
        # return the vector of p(w_i|c_0), p(w_i|c_1)
        # and p(c_1)
        return p0Vect, p1Vect, pNegative

## 分类器

    def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
        p1 = sum(vec2Classify*p1Vec)+np.log(pClass1)
        p0 = sum(vec2Classify*p0Vec)+np.log(1-pClass1)
        if p1 > p0:
            return 1
        else:
            return 0

## 对语段的预处理

    def textParse(bigString):
        import re
        listOfTokens = re.split(r'\w*', bigString)
        return [tok.lower() for tok in listOfTokens if len(tok) > 2]

## 测试

    def testNB():
        listOPosts, listClasses = loadDataSet()
        myVocabList = createVocabList(listOPosts)
        # create a matrix of 0/1 vectors
        trainMatrix = []
        for postinDoc in listOPosts:
            trainMatrix.append(bagOfWords2Vec(myVocabList, postinDoc))\
                # calculate p1,p0
        p0V, p1V, pNe = trainNB0(trainMatrix, listClasses)
        testEntry = ['love', 'my', 'dalmation']
        thisDoc = np.array(bagOfWords2Vec(myVocabList, testEntry))
        print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pNe))
        testEntry = ['stupid', 'garbage']
        thisDoc = np.array(bagOfWords2Vec(myVocabList, testEntry))
        print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pNe))

    testNB()
