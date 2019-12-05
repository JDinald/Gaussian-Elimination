import numpy as np
import sys
import time

matrixAB = np.loadtxt('matrix.txt')
B = np.copy(matrixAB[:, matrixAB.shape[1] - 1])


def foo(matrix):
    start=time.process_time() 
    AB = np.copy(matrix)
    numOfRows = AB.shape[0]
    numOfColumns = AB.shape[1] - 1
    columnNum = 0
    xLst = []

    """"Lead element search"""
    print("Matrix before leading coefficient search: ")
    print(AB)
    print(" ")
    AB[[0, np.argmax(abs(AB[:, 0]))]] = AB[[np.argmax(abs(AB[:, 0])), 0]]
    if AB[0, 0] == 0.0:
        sys.exit("Matrix is not correct")

    print("Matrix after leading coefficient search: ")
    print(AB)
    print(" ")

    """Upper triangular matrix"""
    while columnNum < numOfColumns:
        columnNum += 1
        for i in range(columnNum, numOfRows):
            AB[i, :] = AB[i, :] - AB[i, columnNum - 1] / AB[columnNum - 1, columnNum - 1] * AB[columnNum - 1, :]

    print("Upper triangular matrix: ")
    print(AB)
    print(" ")

    """Find x vector"""

    while columnNum != 0:
        columnNum -= 1
        lineOfX = AB[columnNum, numOfRows]
        if columnNum + 1 != numOfRows:
            for y in range(1, numOfRows - columnNum):
                lineOfX += -AB[columnNum, numOfRows - y] * xLst[y - 1]
        x = lineOfX / AB[columnNum, columnNum]
        xLst.append(x)

    stop = time.process_time() 
    xLst.reverse()
    print("x vector: ")
    print(xLst)
    print(" ")
    print("Start time: ", start, "End time: ", stop)
    print("Elapsed time during the whole function in seconds:", stop-start)
    return np.asarray(xLst)

vectorOfXAlpha = foo(matrixAB)

"""Cond(A)"""
modifiedB = np.copy(B)
modifiedB[np.argmax(abs(B))] = B[np.argmax(abs(B))] / 100 * 101

matrixAB[:, matrixAB.shape[1] - 1] = modifiedB
print()
print("Cond(A) check: ")
vectorOfXBeta = foo(matrixAB)

deltaB = modifiedB - B
deltaX = vectorOfXAlpha - vectorOfXBeta
print(" ")
condA = abs(np.sum(deltaX) / np.sum(vectorOfXAlpha)) * (np.sum(B) / np.sum(deltaB))
print("Cond(A) =< {:03f}".format(condA))
