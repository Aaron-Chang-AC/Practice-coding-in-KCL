import numpy as np

def karhunen_loweve_transform(x):
    """

    :param a: array
    :return:
    """
    mu = np.mean(x, axis=0)
    diff = x - mu
    cov_matrix = []
    for i in range(len(x)):
        cov_matrix.append(np.outer(diff[i], diff[i].T))
    cov_matrix = (1/len(x))*np.sum(cov_matrix, axis=0)
    val, vec = np.linalg.eig(cov_matrix)
    sort_order= np.argsort(val)[::-1]

    for i in range(len(vec)):
        vec[i] = vec[i][sort_order]

    reduced_list = np.delete(vec, len(vec[0])-1, 1)


    for i in range(len(x)):
        print(np.dot(reduced_list.T, diff[i]))
    return val, vec


a= [[1, 2, 1],[2, 3, 1],[3, 5, 1],[2, 2, 1]]
b= [[0, 1], [3, 5],[5,4], [5,6],[8,7],[9,7]]

klt, vec = karhunen_loweve_transform(b)
# print(klt)
# print(vec)
# print(val)