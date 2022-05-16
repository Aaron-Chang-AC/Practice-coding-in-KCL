import numpy as np
from itertools import combinations

def heaviside_function(wx, H_0):
    result = 0.0
    if wx < (10 ** (-9)) and wx > (10 ** (-9) * (-1)):
        result = H_0
    elif wx > 0:
        result = 1
    else:
        result = 0

    return result

def PCA(S, dimension, new_samples_to_be_classified,input_eigenvectors=[],input_eigenvalues=[]):
    n = len(S)
    X = S.copy().T
    print(f"original dataset(each column is a sample):\n{X}\n")
    m = X.mean(axis=1)
    print(f"mean of X:\n{m.reshape(-1,1)}\n")
    Xi_m= X-m.reshape(-1,1)
    print(f"zero mean dataset:\n{Xi_m}\n")
    covariance_matrix=(Xi_m @ Xi_m.T)/n
    print(f"covariance_matrix:\n{covariance_matrix}\n")
    # V, D, VT = np.linalg.svd(covariance_matrix, full_matrices=True)
    D, V = np.linalg.eigh(covariance_matrix)
    if len(input_eigenvalues) > 0:
        D = input_eigenvalues
        V = input_eigenvectors
    idx = (-D).argsort()
    D = np.diag(D[idx])
    V = V[:,idx]
    print(f"V:\n{V}\n")
    print(f"D:\n{D}\n")
    print(f"VT:\n{V.T}\n")

    V_hat = V.copy()[:,0:dimension]
    print(f"V_hat:\n{V_hat}\n")
    V_hatT = V_hat.T
    print(f"V_hatT:\n{V_hatT}\n")
    results = V_hatT @ Xi_m
    print(f"results(each column is a converted sample):\n{results}\n")

    if len(new_samples_to_be_classified)>0:
        new_targets=V_hatT @ (new_samples_to_be_classified.T - m.reshape(-1,1))
        print(f"new_samples_to_be_classified(each column is a converted sample):\n{new_targets}\n")

    return results

def oja_learning(datapoints,given_zero_mean=False, initial_weight=None,learning_rate = 0.01, epoch=2, mode=None):
    n = len(datapoints)
    X = datapoints.copy().T
    print(f"original dataset(each column is a sample):\n{X}\n")
    Xi_m = X
    if not given_zero_mean:
        m = X.mean(axis=1)
        print(f"mean of X:\n{m.reshape(-1, 1)}\n")
        Xi_m = X - m.reshape(-1, 1)
    print(f"zero mean datapoints(each column is a sample):\n{Xi_m}\n")
    for i in range(epoch):
        print(f"---------Epoch {i+1} start---------")
        if mode == "batch":
            total_weight_change=np.zeros(len(initial_weight))
            for j in range(n):
                # y = initial_weight @ datapoints.T[j]
                # print(initial_weight, Xi_m.T[j])
                y = np.dot(initial_weight, Xi_m.T[j])
                print(f"y is {y}")
                xt_yw = Xi_m.T[j] - (y * initial_weight)
                weight_change = learning_rate * (y * xt_yw)
                total_weight_change+=weight_change
                print(f"xt:{Xi_m.T[j]}\ny=wx:{y}\nxt-yw:{xt_yw}\nphi*y(xt-yw):{weight_change}\n\n")
            print(f"total weight change:{total_weight_change}")
            initial_weight = np.add(initial_weight,total_weight_change)
            print(f"new weight:{initial_weight}\n")
        else:
            for j in range(n):
                y = np.dot(initial_weight, Xi_m.T[j])
                print(f"y is {y}")
                xt_yw = Xi_m.T[j] - (y * initial_weight)
                weight_change = learning_rate * (y * xt_yw)
                initial_weight= np.add(initial_weight, weight_change)
                print(f"xt:{Xi_m.T[j]}\ny=wx:{y}\nxt-yw:{xt_yw}\nphi*y(xt-yw):{weight_change}\n\n")
            print(f"new weight:{initial_weight}\n")

    return initial_weight


def hebbian_learning(datapoints, given_zero_mean=False, initial_weight=None,learning_rate = 0.01, epoch=2, mode=None):
    n = len(datapoints)
    X = datapoints.copy().T
    print(f"original dataset(each column is a sample):\n{X}\n")
    Xi_m = X
    if not given_zero_mean:
        m = X.mean(axis=1)
        print(f"mean of X:\n{m.reshape(-1, 1)}\n")
        Xi_m = X - m.reshape(-1, 1)
    print(f"zero mean datapoints(each column is a sample):\n{Xi_m}\n")
    for i in range(epoch):
        print(f"---------Epoch {i+1} start---------")
        if mode == "batch":
            total_weight_change=np.zeros(len(initial_weight))
            for j in range(n):
                # y = initial_weight @ datapoints.T[j]
                # print(initial_weight, Xi_m.T[j])
                y = np.dot(initial_weight, Xi_m.T[j])
                print(f"y is {y}")
                # xt_yw = Xi_m.T[j] - (y * initial_weight)
                weight_change = learning_rate * (y * Xi_m.T[j] )
                total_weight_change+=weight_change
                print(f"xt:\n{Xi_m.T[j]}\ny=wx:\n{y}\nphi*y(xt):\n{weight_change}")
                print(f"--------------------------")
            print(f"total weight change:\n{total_weight_change}")
            initial_weight = np.add(initial_weight,total_weight_change)
            print(f"new weight:\n{initial_weight}\n")
        else:
            for j in range(n):
                y = np.dot(initial_weight, Xi_m.T[j])
                print(f"y is {y}")
                # xt_yw = Xi_m.T[j] - (y * initial_weight)
                weight_change = learning_rate * (y * Xi_m.T[j])
                initial_weight= np.add(initial_weight, weight_change)
                print(f"xt:\n{Xi_m.T[j]}\ny=wx:\n{y}\nphi*y(xt):\n{weight_change}")
                print(f"new weight:\n{initial_weight}\n")
                print(f"--------------------------")
            print(f"new weight:\n{initial_weight}\n")

    return initial_weight

def fisher_method(datapoints, classes, projection_weight):
    class_set = set(sorted(classes.tolist()))
    number_classes = len(class_set)
    number_datapoints = len(datapoints)
    means = np.zeros((number_classes,len(datapoints[0])))
    counter = np.zeros(number_classes)

    print(class_set)
    
    for i in range(number_datapoints):
        means[classes[i]-1] = np.add(means[classes[i]-1],datapoints[i])
        counter[classes[i]-1] = np.add(counter[classes[i]-1], 1)
    for i in range(number_classes):
        means[i] = means[i]/counter[i]
        print(f"mean {i+1}: {means[i]}\n")

    cost = np.zeros(len(projection_weight))

    for i in range(len(projection_weight)):
        sb = 0.0
        comb = list(combinations(list(class_set), 2))
        # print(comb)
        for j in range(len(comb)):
            sb = sb + np.square(projection_weight[i] @ (np.subtract(means[comb[j][0]-1], means[comb[j][1]-1])))

        print(f"sb for w{i+1}: {sb}")
        
        sw = 0.0
        for j in range(len(datapoints)):
            sw += np.square(projection_weight[i] @ (np.subtract(datapoints[j], means[classes[j]-1])))

        print(f"sw for w{i+1}: {sw}")

        cost[i] = sb/sw
        print(f"Cost of w{i+1} is {cost[i]}\n")

    winner = np.argmax(cost)
    print(f"Effective projection weight is {projection_weight[winner]}")
    return projection_weight[winner]

def extreme_learning_machine(V, weight, datapoints, H_0):
    datapoints_T = datapoints.T.copy()
    new_datapoints = np.ones((len(datapoints),1))
    new_datapoints = np.append(new_datapoints, datapoints, axis=1)
    new_datapoints = new_datapoints.T
    print(f"augmented new_datapoints:\n{new_datapoints}\n")
    VX = V @ new_datapoints
    print(f"V*X:\n{VX}\n")
    f = np.vectorize(heaviside_function)
    Y = f(VX, H_0)
    print(f"Y = H(VX):\n{Y}\n")
    new_Y = np.ones((len(Y.T), 1))
    new_Y = np.append(new_Y, Y.T, axis=1)
    new_Y = new_Y.T
    Z = weight @ new_Y
    print(f"The final response of the output neuron:\n{Z}")

def sparse_coding(V, x, y_t, LAMBDA = 0):
    error_result = np.zeros(len(y_t))
    for i in range(len(y_t)):
        error = np.subtract(x.T, (V.T @ y_t[i].T))
        error = np.linalg.norm(error) + LAMBDA * np.linalg.norm(y_t[i],ord=0)
        error_result[i]=error
        print(f"error for y_t{i+1}: {error}")
    print(f"best sparse code: y_t{np.argmin(error_result)+1}")

def LDA(datapoints, classes, projection_vector):
    for i in range(len(datapoints)):
        y = projection_vector @ datapoints[i]
        print(f"y = w_T * x = {y}")
        print(f"The belonging class is: {classes[i]}")

    print("Check with your common sense if it is obviously separable")
    return

# EXECUTION ###################################################
# for PCA ####################################
# Note that new_samples_to_be_classified has not subtracted by the mean yet
# S = np.asarray([
#     [1,2,1],
#     [2,3,1],
#     [3,5,1],
#     [2,2,1]
# ], dtype=np.float32)
# new_samples_to_be_classified = np.asarray([
#         [1,2,1]
# ])
#
# results = PCA(S,dimension=2,new_samples_to_be_classified=new_samples_to_be_classified)
# print(results)


# for PCA given eigenvalues and eigenvectors
# Note that new_samples_to_be_classified has not subtracted by the mean yet
# input_eigenvalues = np.asarray([0,0.71,1.9,3.21],dtype=np.float32)
# input_eigenvectors = np.asarray([
#     [-0.59,0.55,0.11,0.58],
#     [-0.56,-0.78,0.25,0.12],
#     [0.25,0.12,0.96,-0.04],
#     [0.52,-0.27,-0.07,0.81]
# ])
# S = np.asarray([
#     [5.0, 5.0, 4.4, 3.2],
#     [6.2, 7.0, 6.3, 5.7],
#     [5.5, 5.0, 5.2, 3.2],
#     [3.1, 6.3, 4.0, 2.5],
#     [6.2, 5.6, 2.3, 6.1]
# ])
# new_samples_to_be_classified = np.asarray([
#     [5.0, 5.0, 4.4, 3.2]
# ])
#
# results = PCA(S,dimension=2,new_samples_to_be_classified=new_samples_to_be_classified,
#               input_eigenvalues=input_eigenvalues,
#               input_eigenvectors=input_eigenvectors
#               )
# print(results)


# for oja's learning rule ####################################
# Note that given_zero_mean is a flag which should be true when given
# dataset is zero mean
# S = np.asarray([
#     [0,1],
#     [3,5],
#     [5,4],
#     [5,6],
#     [8,7],
#     [9,7]
# ],dtype=np.float32)
# oja_learning(datapoints=S, given_zero_mean=False, initial_weight=np.asarray([-1, 0]), learning_rate=0.01, epoch=2, mode="batch")
# oja_learning(datapoints=S, initial_weight=np.asarray([-1, 0]), learning_rate=0.01, epoch=6, mode="sequential")

# for Fisher’s method (LDA) ####################################
# datapoints = np.asarray([[1, 2], [2, 1],[3, 3], [6, 5], [7, 8]])
# classes = np.asarray([1, 1, 1, 2, 2])
# projection_weight = np.asarray([[-1, 5], [2, -3], [1 , 1]])
# fisher_method(datapoints, classes, projection_weight=projection_weight)

# for extreme learning machine ####################################
# datapoints = np.asarray([[0,0], [0, 1],[1, 0], [1, 1]])
# weight = np.asarray([0,0,0,-1,0,0,2])
# V = np.asarray([[-0.62, 0.44, -0.91],
#                 [-0.81, -0.09, 0.02],
#                 [0.74, -0.91, -0.60],
#                 [-0.82, -0.92, 0.71],
#                 [-0.26, 0.68, 0.15],
#                 [0.8, -0.94, -0.83]])
# extreme_learning_machine(V, weight, datapoints, H_0=0.5)

# for extreme learning machine ####################################
# datapoints = np.asarray([[0,0], [0, 1],[1, 0], [1, 1]])
# weight = np.asarray([0,0,0,-1,0,0,2])
# V = np.asarray([[-0.62, 0.44, -0.91],
#                 [-0.81, -0.09, 0.02],
#                 [0.74, -0.91, -0.60],
#                 [-0.82, -0.92, 0.71],
#                 [-0.26, 0.68, 0.15],
#                 [0.8, -0.94, -0.83]])
# extreme_learning_machine(V, weight, datapoints, H_0=0.5)

# for sparse_coding ###################################
# Note that LAMBDA is the coefficient for L0 norm
# V = np.asarray([[0.4, -0.6],
#               [0.55, -0.45],
#               [0.5, -0.5],
#               [-0.1, 0.9],
#               [-0.5, -0.5],
#               [0.9, 0.1],
#               [0.5, 0.5],
#               [0.45, 0.55]])
# x = np.asarray([-0.05, -0.95])
# y1_t = np.asarray([1, 0, 0, 0, 1, 0, 0, 0])
# y2_t = np.asarray([0, 0, 1, 0, 0, 0, -1, 0])
# y3_t = np.asarray([0, 0, 0, -1, 0, 0, 0, 0])
# y_t = np.asarray([y1_t, y2_t, y3_t])
# sparse_coding(V, x, y_t, LAMBDA=0.0)

# for sparse_coding with LAMBDA != 0.0 ######################
# Note that LAMBDA is the coefficient for L0 norm
# V = np.asarray([
#     [1, -4],
#     [1, 3],
#     [2, 2],
#     [1, -1]
# ], dtype = np.float32)
# x = np.asarray([2.0, 3.0])
#
# y1_t = np.asarray([1, 2, 0, -1])
# y2_t = np.asarray([0, 0.5, 1, 0])
#
# y_t = np.asarray([y1_t, y2_t])
# sparse_coding(V, x, y_t, LAMBDA=1.0)

# for LDA
# datapoints = np.array([
#     [1, 2],
#     [2, 1],
#     [3, 3],
#     [6, 5],
#     [7, 8]
# ],dtype=np.int8)
#
# classes = np.array([1, 1, 1, 2, 2])
# projection_vector = np.array([1, 2],dtype=np.int8)
# LDA(datapoints=datapoints, classes= classes, projection_vector=projection_vector)

# for hebbian_learning
# Note that given_zero_mean is a flag which should be true when given
# dataset is zero mean
# S = np.asarray([
#     [0,1],
#     [3,5],
#     [5,4],
#     [5,6],
#     [8,7],
#     [9,7]
# ],dtype=np.float32)
# initial_weight=np.asarray([-1, 0],dtype=np.float32)
# hebbian_learning(datapoints=S, given_zero_mean=True, initial_weight=initial_weight, learning_rate=0.01, epoch=6, mode="batch")