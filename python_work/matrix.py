import numpy as np


i = 1
matrix1 = []
matrix2 = []
while True:
    output = 'enter the '+str(i)+'\'th row of the lhs matrix: '
    string = input(output)
    if len(string) == 0:
        break
    i = i+1
    curLine = []
    string = string.split(',')
    for j in range(len(string)):
        curLine.append(float(string[j]))
    matrix1.append(curLine)
i = 1
while True:
    output = 'enter the '+str(i)+'\'th row of the rhs matrix: '
    string = input(output)
    if len(string) == 0:
        break
    i = i+1
    curLine = []
    string = string.split(',')
    for j in range(len(string)):
        curLine.append(float(string[j]))
    matrix2.append(curLine)

matrix1 = np.mat(matrix1)
matrix2 = np.mat(matrix2)

retMatrix = np.dot(matrix1, matrix2)

print(retMatrix)
