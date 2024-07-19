import numpy as np


def isDiagonallyDominant(matrix):
    n = len(matrix)

    # Check for row diagonal dominance
    for i in range(n):
        row_sum = sum(abs(matrix[i][j]) for j in range(n) if j != i)
        if abs(matrix[i][i]) < row_sum:
            return False

    # Check for column diagonal dominance
    for j in range(n):
        col_sum = sum(abs(matrix[i][j]) for i in range(n) if i != j)
        if abs(matrix[j][j]) < col_sum:
            return False

    return True


def jacobi_method(A, b, tolerance):
    n = len(A)
    x = np.zeros(n)
    x_new = np.zeros(n)

    if not isDiagonallyDominant(A):
        print("Matrix is not diagonally dominant")
        return None

    while True:
        for i in range(n):
            sum = b[i]
            for j in range(n):
                if i != j:
                    sum -= A[i][j] * x[j]
            x_new[i] = sum / A[i][i]

        # Check for convergence
        if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
            return x_new

        x = x_new.copy()


def gauss_seidel_method(A, b, tolerance):
    n = len(A)
    x = np.zeros(n)

    if not isDiagonallyDominant(A):
        print("Matrix is not diagonally dominant")
        return None

    while True:
        x_new = x.copy()
        for i in range(n):
            sum = b[i]
            for j in range(n):
                if i != j:
                    sum -= A[i][j] * x_new[j]
            x_new[i] = sum / A[i][i]

        # Check for convergence
        if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
            return x_new

        x = x_new


# Example usage:
def main():
    # Definition of the tolerance
    epsilon = 0.001

    # Definition of the matrix of coefficients A
    A = [
        [5, 2, 1],
        [1, 6, 2],
        [2, 1, 4]
    ]

    # Defining vector B for example
    b = [12, 19, 16]

    print("Jacobi method solution:")
    jacobi_solution = jacobi_method(A, b, epsilon)
    if jacobi_solution is not None:
        print(jacobi_solution)

    print("\nGauss-Seidel method solution:")
    gauss_seidel_solution = gauss_seidel_method(A, b, epsilon)
    if gauss_seidel_solution is not None:
        print(gauss_seidel_solution)


if __name__ == "__main__":
    main()
