THEOREMQA_PROMPT_FS = """
The following are three examples for reference.

The question is : Suppose that $X_1,X_2,...$ are real numbers between 0 and 1 that are chosen independently and uniformly at random. Let $S=\\sum_{i=1}^k X_i/2^i$, where $k$ is the least positive integer such that $X_k<X_{k+1}$, or $k=\\infty$ if there is no such integer. Find the expected value of S.
The solution code is:
```python
def solution():
    '''Suppose that $X_1,X_2,...$ are real numbers between 0 and 1 that are chosen independently and uniformly at random. Let $S=\\sum_{i=1}^k X_i/2^i$, where $k$ is the least positive integer such that $X_k<X_{k+1}$, or $k=\\infty$ if there is no such integer. Find the expected value of S.'''
    import numpy as np

    # Number of simulations
    n_simulations = 1000000

    # Initialize the sum
    total_sum = 0

    for _ in range(n_simulations):
        k = 1
        X_prev = np.random.uniform(0, 1)
        X_next = np.random.uniform(0, 1)

        # Find the least positive integer k such that X_k < X_{k+1}
        while X_prev >= X_next:
            total_sum += X_prev / (2 ** k)
            k += 1
            X_prev = X_next
            X_next = np.random.uniform(0, 1)

        # Add the last term
        total_sum += X_prev / (2 ** k)

    # Calculate the expected value
    expected_value = total_sum / n_simulations
    result = expected_value
    return result
```

The question is : For the two linear equations $2 * x + 3 * y + z = 8$ and $4 * x + 4 * y + 4z = 12$ and $x + y + 8z = 10$ with variables x, y and z. Use cramer's rule to solve these three variables.
The solution code is:
```python
def solve():
    '''For the two linear equations $2 * x + 3 * y + z = 8$ and $4 * x + 4 * y + 4z = 12$ and $x + y + 8z = 10$ with variables x, y and z. Use cramer's rule to solve these three variables.'''

    # Coefficient matrix
    A = np.array([[2, 3, 1], [4, 4, 4], [1, 1, 8]])

    # Constants matrix
    B = np.array([8, 12, 10])

    # Check if the determinant of A is non-zero
    det_A = np.linalg.det(A)
    if det_A == 0:
        return "No unique solution"

    # Calculate the determinants of the matrices obtained by replacing the columns of A with B
    A_x = A.copy()
    A_x[:, 0] = B
    det_A_x = np.linalg.det(A_x)

    A_y = A.copy()
    A_y[:, 1] = B
    det_A_y = np.linalg.det(A_y)

    A_z = A.copy()
    A_z[:, 2] = B
    det_A_z = np.linalg.det(A_z)

    # Calculate the values of x, y, and z using Cramer's rule
    x = det_A_x / det_A
    y = det_A_y / det_A
    z = det_A_z / det_A

    return [x, y, z]
```

The question is: Does f (x) = x2 + cx + 1 have a real root when c=0?
The solution code is:
```python
def solution():
    '''Does f (x) = x2 + cx + 1 have a real root when c=0?'''
    # When c = 0, the function becomes f(x) = x^2 + 1. To determine if it has a real root, we can look at the discriminant, which is given by the formula D = b^2 - 4ac. In this case, a = 1, b = 0, and c = 1.

    # D = (0)^2 - 4(1)(1) = 0 - 4 = -4

    # Since the discriminant is negative, the function f(x) = x^2 + 1 does not have a real root when c = 0. Therefore, the answer is False.

    result = False
    return result
```

""".strip()