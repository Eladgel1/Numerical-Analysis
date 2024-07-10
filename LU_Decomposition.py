import numpy as np


# Part 1 - LU decomposition
def decomposition_LU (matrix):

    n = len(matrix)
    A = np.array(matrix, dtype=float)

    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):

        for k in range(i, n):
            sum = 0
            for j in range(i):
                sum += (L[i][j] * U[j][k])
            U[i][k] = A[i][k] - sum

        for k in range(i, n):
            if i == k:
                L[i][i] = 1
            else:
                sum = 0
                for j in range(i):
                    sum += (L[k][j] * U[j][i])
                L[k][i] = (A[k][i] - sum) / U[i][i]
    return L, U


# Part 2.1 - Invert matrix
# Code for inverted mat
def is_Invertible(matrix):
    # Check if the matrix is square and has a non-zero determinant.

    m = len(matrix)
    n = len(matrix[0])

    if m == n:
        det = np.linalg.det(matrix)
        return det != 0

    return False


# Code for inverted mat
def InvertMatrix(matrix):
    # Invert a matrix

    if not is_Invertible(matrix):
        raise ValueError("Matrix cannot be inverted.")

    n = len(matrix)
    A = np.array(matrix, dtype=float)
    I = np.eye(n)  # Identity matrix of the same size

    for i in range(n):
        # Check for zero pivot and swap rows if necessary
        if A[i,i] == 0:
            for k in range(i + 1, n):
                if A[k,i] != 0:
                    # Swap rows in both 'A' and 'I'
                    A[[i,k]] = A[[k,i]]
                    I[[i,k]] = I[[k,i]]
                    break
            else:
                raise ValueError("Matrix cannot be inverted.")

        pivot = A[i, i]
        E = np.eye(n)  # Create elementary matrix for scaling
        E[i, i] = 1 / pivot
        A = np.matmul(E,A)
        I = np.matmul(E,I)

        for j in range(n):
            if i != j:
                factor = A[j, i]
                E = np.eye(n)
                E[j, i] = -factor
                A = np.matmul(E, A)
                I = np.matmul(E, I)

    return I.tolist()  # Convert result back to list of lists


# Part 2.2 - Solving a linear equations system using the LU decomposition method
def solve_LU(L, U, b):
    n = L.shape[0]
    c = np.zeros(n)
    for i in range(n):
        c[i] = b[i]
        for k in range(i):
            c[i] -= L[i][k] * c[k]
        c[i] = c[i] / L[i][i]

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = c[i]
        for k in range(i+1, n):
            x[i] -= U[i][k] * x[k]
        x[i] = x[i] / U[i][i]

    return x


# Function for printing the matrix
def print_mat(matrix):
    for row in matrix:
        print(f"{row}".replace(",", " "))


def main():
    # Definition of the matrix of coefficients A
    A = np.array([
        [1, 4, -3],
        [-2, 1, 5],
        [3, 2, 1]
    ])

    print("Matrix A:")
    print(A)

    # Perform LU decomposition
    L, U = decomposition_LU(A)
    print("\nMatrix L:")
    print_mat(L.tolist())

    print("\nMatrix U:")
    print_mat(U.tolist())

    # Finding the invertible matrix of A
    Ainv = InvertMatrix(A)
    print("\nThe invertible matrix of A A:")
    print_mat(Ainv)

    # Defining vector B for example
    b = np.array([1, 2, 3])

    # Solution vector of the system of equations Ax=b using LU decomposition
    x = solve_LU(L, U, b)
    print("\nSolution vector x:")
    print(f"{x}".replace(" ", ", "))


if __name__ == "__main__":
    main()
