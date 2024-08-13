import sys
import numpy as np


# Function for linear interpolation
def linear_interpolation(table_points, x):
    for i in range(len(table_points) - 1):
        x1, y1 = table_points[i]
        x2, y2 = table_points[i + 1]
        # Check if x is between x1 and x2 for interpolation
        if x1 <= x <= x2:
            y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
            return y

    # Extrapolation if the point is outside the range of the table points
    if x < table_points[0][0]:  # Case where x is less than the first x in the table
        x1, y1 = table_points[0]
        x2, y2 = table_points[1]
    else:  # Case where x is greater than the last x in the table
        x1, y1 = table_points[-2]
        x2, y2 = table_points[-1]
    y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    return y


# Function for polynomial interpolation
def polynomial_interpolation(table_points, x):
    n = len(table_points)

    matrix = [[point[0] ** i for i in range(n)] for point in table_points]
    b = [point[1] for point in table_points]  # Create the b vector

    # Solve the linear system to find the coefficients
    coefficients = np.linalg.solve(matrix, b)

    # Calculate the polynomial result for the given x
    result = sum(coefficients[i] * (x ** i) for i in range(n))
    return result


# Function for Lagrange interpolation
def lagrange_interpolation(table_points, x):
    result = 0
    # Iterate over each point to construct Lagrange polynomial
    for i in range(len(table_points)):
        xi, yi = table_points[i]
        li = 1
        for j in range(len(table_points)):
            if i != j:
                xj, _ = table_points[j]
                li *= (x - xj) / (xi - xj)  # Calculate the Lagrange basis polynomial
        result += li * yi  # Add the contribution of each term to the result
    return result


def main():
    table_points = [(0, 0), (1, 0.7451), (2, 1.2744), (3, -0.4392), (4, -0.7219), (5, 0.5616), (6, 1.4723)]
    x = 1.17

    print("\nTable Points:", table_points)
    print("\nFinding an approximation of the value:", x)

    # Get the user's choice of interpolation method
    choice = int(input("\nChoose one the following methods: \n\t"
                       "1. Linear Method \n\t"
                       "2. Polynomial Method\n\t"
                       "3. Lagrange Method\n"))

    # Perform the interpolation based on the user's choice
    match choice:
        case 1:
            y = linear_interpolation(table_points, x)
        case 2:
            y = polynomial_interpolation(table_points, x)
        case 3:
            y = lagrange_interpolation(table_points, x)
        case _:
            sys.exit("Invalid input")

    # Print the result
    print(f"The approximation of the value {x} is: {round(y, 4)}")


# Entry point of the program
if __name__ == "__main__":
    main()
