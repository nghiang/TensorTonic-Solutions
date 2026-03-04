import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    AT = np.zeros((len(A[0]), len(A)))
    for i in range(0, len(A)):
        for j in range(0, len(A[0])):
            AT[j][i]=A[i][j]
    return AT
