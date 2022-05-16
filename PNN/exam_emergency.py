import numpy as np




A = np.array([
    [2, 1],
    [1, 4]
])

b = np.array([
    [1, 2]
])

c = -3

x_T = np.array([
    [0, -1],
    [1, 1]
])

for i in range(len(x_T)):
    result = x_T[i] @ A @ x_T[i].T + x_T[i] @ b.T +c
    print(result)