import numpy as np
import matplotlib.pyplot as plt

def show_points(datapoint, classes):
    datapoint=np.asarray(datapoint)
    classes = np.asarray(classes)

    plt.scatter(datapoint[:,0],datapoint[:,1], c = classes)
    plt.show()
    return

def LinearSVM(svs, classes):
    """

    :param svs: list of support vectors
    :param classes: list
    :return:
    """
    support_vectors = np.asarray(svs, dtype=np.float32)
    y = np.asarray(classes, dtype=np.int8)

    last_line = y.copy().tolist()
    last_line.append(0)

    a = []
    for i in range(len(support_vectors)):
        a.append([])
        for j in range(len(support_vectors)):
            a[i].append(y[j] * np.matmul(support_vectors[j], support_vectors[i].T))
        a[i].append(1)
    a.append(last_line)
    A = np.asarray(a)
    print(A)

    # pseudo inverse
    A_inv = np.linalg.pinv(A)
    res = np.matmul(A_inv, np.asarray(last_line)).flatten()

    for i in range(len(support_vectors)):
        print("lambda ", i + 1, ": ", res[i])
    print("w0: ", res[-1])

    w = np.zeros(len(support_vectors[0]))
    for i in range(len(support_vectors)):
        w = w + y[i] * round(res[i],4) * support_vectors[i]
    hyperplane = np.zeros(len(support_vectors[0]) + 1)
    print("w:\n", w.T)
    hyperplane[0:len(support_vectors[0])] = w[0:len(support_vectors[0])].copy()
    hyperplane[-1] = res[-1]
    print("hyperplane(the first term is w1 and the last one is w0): \n", hyperplane)
    return hyperplane

def margin_from_hyperplane(hyperplane):
    # print(hyperplane[:-1])
    # print(np.linalg.norm(hyperplane[:-1]))
    margin = 2 / (np.linalg.norm(hyperplane[:-1]))
    print(f"Margin is: {margin}")
    return margin

# EXECUTION ##################
# svs = [
#     [1,1],
#     [1,-1],
#     [-1,1],
#     [-1,-1]
# ]
# classes = [1,1,-1,-1]
# LinearSVM(svs, classes)

# svs = [
#     [-1, 0],
#     [1, 0]
# ]
# classes = [1,-1]
# LinearSVM(svs, classes)

# To plot the dataset
# svs = [
#     [1,1],
#     [1,-1],
#     [-1,1],
#     [-1,-1]
# ]
# classes = [1,1,-1,-1]
# show_points(svs,classes)