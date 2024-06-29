import numpy as np


def calc_cond(matrix):
    # Calculate the condition number of the matrix.

    mat_1 = InvertMatrix(matrix)  # The inverse matrix
    return matNormal(matrix) * matNormal(mat_1)


def is_Invertible(matrix):
    # Check if the matrix is square and has a non-zero determinant.

    m = len(matrix)
    n = len(matrix[0])

    if m == n:
        det = np.linalg.det(matrix)
        return det != 0

    return False


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
                    # Swap rows in both A and I
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
                A = np.matmul(E,A)
                I = np.matmul(E,I)

    return I.tolist()  # Convert result back to list of lists


def matNormal(matrix):
    # Calculate the infinity norm (maximum absolute row sum) of the matrix.

    if is_Invertible(matrix):
        rows_sum = []
        for row in matrix:
            row_abs = [abs(element) for element in row]
            rows_sum.append(sum(row_abs))

        return max(rows_sum)
    else:
        raise ValueError("Matrix cannot be inverted.")


def print_mat(matrix):
    for row in matrix:
        print(f"{row}".replace(",", " "))


# Example usage:
mat = [[1, -1, -2], [2, -3, -5], [-1, 3, 5]]

print("Matrix A:")
print_mat(mat)

print("\nInverse Matrix:")
print_mat(InvertMatrix(mat))

condition_number = calc_cond(mat)
print(f"\nFinal condition number: {condition_number}")