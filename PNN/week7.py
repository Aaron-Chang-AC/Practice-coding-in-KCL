import numpy as np

def heaviside_function(wx, H_0):
    result = 0.0
    if wx < (10 ** (-9)) and wx > (10 ** (-9) * (-1)):
        result = H_0
    elif wx > 0:
        result = 1
    else:
        result = 0

    return result

def PCA(S, dimension, new_samples_to_be_classified):
    n = len(S)
    X = S.copy().T
    print(f"original dataset(each column is a sample):\n{X}\n")
    m = X.mean(axis=1)
    Xi_m= X-m.reshape(-1,1)
    covariance_matrix=(Xi_m @ Xi_m.T)/n
    print(f"covariance_matrix:\n{covariance_matrix}\n")
    V, D, VT = np.linalg.svd(covariance_matrix, full_matrices=True)
    D = np.diag(D)
    print(f"V:\n{V}\n")
    print(f"D:\n{D}\n")
    print(f"VT:\n{VT}\n")

    V_hat = V.copy()[:,0:dimension]
    V_hatT = V_hat.T
    print(f"V_hatT:\n{V_hatT}\n")
    results = V_hatT @ Xi_m
    print(f"results(each column is a converted sample):\n{results}\n")

    if len(new_samples_to_be_classified)>0:
        new_targets=V_hatT @ new_samples_to_be_classified.T
        print(f"results(each column is a converted sample):\n{new_targets}\n")

    return results

# Dr Michael Version
def KL_Transform(S, dimension, new_samples_to_be_classified):
    n = len(S)
    X = S.copy().T
    print(f"original dataset(each column is a sample):\n{X}\n")
    m = X.mean(axis=1)
    Xi_m = X - m.reshape(-1, 1)
    covariance_matrix = (Xi_m @ Xi_m.T) / n
    print(f"covariance_matrix:\n{covariance_matrix}\n")
    W, V = np.linalg.eigh(covariance_matrix) # W is eigenvalues in ascending order and V is the normalized eigenvector
    # sort in descending for both eigenvalues and eigenvectors
    idx = W.argsort()[::-1]
    W = W[idx]
    V = V[:,idx]

    print(f"Eigenvalues (Sorted):\n{W}\n")
    print(f"Eigenvectors(Sorted):\n{V}\n")

    V_hat = V.copy()[:, 0:dimension]
    V_hatT = V_hat.T
    print(f"V_hatT:\n{V_hatT}\n")
    results = V_hatT @ Xi_m
    print(f"results(each column is a converted sample):\n{results}\n")

    if len(new_samples_to_be_classified) > 0:
        new_targets = V_hatT @ new_samples_to_be_classified.T
        print(f"results(each column is a converted sample):\n{new_targets}\n")

    return results


# S = np.asarray([
#     [1,2,1],
#     [2,3,1],
#     [3,5,1],
#     [2,2,1]
# ])
# new_samples_to_be_classified = np.asarray([
#         [3,-2,5]
# ])
# print(KL_Transform(S=S, dimension=2, new_samples_to_be_classified=new_samples_to_be_classified))

# S = np.asarray([
#     [0,1],
#     [3,5],
#     [5,4],
#     [5,6],
#     [8,7],
#     [9,7]
# ])

# results = PCA(S,dimension=2,new_samples_to_be_classified=[])
# # print(results)
# print(PCA(results.T, dimension=2, new_samples_to_be_classified=[]))
#

def hebbian_learning(datapoints, initial_weight,learning_rate = 0.01, epoch=2, mode=None):
    n = len(datapoints)
    X = datapoints.copy().T
    print(f"original dataset(each column is a sample):\n{X}\n")
    m = X.mean(axis=1)
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
                print(f"xt:\n{Xi_m.T[j]}\ny=wx:\n{y}\nphi*y(xt-yw):\n{weight_change}")
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
                print(f"xt:\n{Xi_m.T[j]}\ny=wx:\n{y}\nphi*y(xt-yw):\n{weight_change}")
            print(f"new weight:\n{initial_weight}\n")

    return initial_weight



def oja_learning(datapoints, initial_weight,learning_rate = 0.01, epoch=2, mode=None):
    n = len(datapoints)
    X = datapoints.copy().T
    print(f"original dataset(each column is a sample):\n{X}\n")
    m = X.mean(axis=1)
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
                print(f"xt:\n{Xi_m.T[j]}\ny=wx:\n{y}\nxt-yw:\n{xt_yw}\nphi*y(xt-yw):\n{weight_change}")
                print(f"--------------------------")
            print(f"total weight change:\n{total_weight_change}")
            initial_weight = np.add(initial_weight,total_weight_change)
            print(f"new weight:\n{initial_weight}\n")
        else:
            for j in range(n):
                y = np.dot(initial_weight, Xi_m.T[j])
                print(f"y is {y}")
                xt_yw = Xi_m.T[j] - (y * initial_weight)
                weight_change = learning_rate * (y * xt_yw)
                initial_weight= np.add(initial_weight, weight_change)
                print(f"xt:\n{Xi_m.T[j]}\ny=wx:\n{y}\nxt-yw:\n{xt_yw}\nphi*y(xt-yw):\n{weight_change}")
            print(f"new weight:\n{initial_weight}\n")

    return initial_weight


# print(oja_learning(datapoints=S, initial_weight=np.asarray([-1, 0]), learning_rate=0.01, epoch=6, mode="batch"))


def fisher_linear_discriminant_analysis(datapoints, classes, projection_weight):
    number_classes = int(np.max(classes))
    number_datapoints = len(datapoints)
    means = np.zeros((number_classes,len(datapoints[0])))
    counter = np.zeros(number_classes)
    for i in range(number_datapoints):
        means[classes[i]-1] = np.add(means[classes[i]-1],datapoints[i])
        counter[classes[i]-1] = np.add(counter[classes[i]-1], 1)
    for i in range(number_classes):
        means[i] = means[i]/counter[i]
        print(f"mean {i+1}:\n{means[i]}\n")
    print(means)


    cost = np.zeros(len(projection_weight))

    for i in range(len(projection_weight)):
        sb = np.square(projection_weight[i] @ (np.subtract(means[0], means[1])))
        sw = 0.0
        for j in range(len(datapoints)):
            if (classes[j]-1) == 0:
                sw += np.square(projection_weight[i] @ (np.subtract(datapoints[j], means[0])))
            else:
                sw += np.square(projection_weight[i] @ (np.subtract(datapoints[j], means[1])))
        print(sb)
        cost[i] = sb/sw
    print(f"Cost of w1 is {cost[0]} and w2 is {cost[1]}")

    winner = np.argmax(cost)
    print(f"Effective projection weight is {projection_weight[winner]}")
    return projection_weight[winner]



# datapoints = np.asarray([[1, 2], [2, 1],[3, 3], [6, 5], [7, 8]])
# classes = np.asarray([1, 1, 1, 2, 2])
# projection_weight = np.asarray([[-1, 5], [2, -3]])
# fisher_linear_discriminant_analysis(datapoints, classes, projection_weight=projection_weight)
# datapoints = np.asarray([[0,0], [0, 1],[1, 0], [1, 1]])
# weight = np.asarray([0,0,0,-1,0,0,2])
# V = np.asarray([[-0.62, 0.44, -0.91],
#                 [-0.81, -0.09, 0.02],
#                 [0.74, -0.91, -0.60],
#                 [-0.82, -0.92, 0.71],
#                 [-0.26, 0.68, 0.15],
#                 [0.8, -0.94, -0.83]])
def extreme_learning_machine(V, weight, datapoints, H_0):
    datapoints_T = datapoints.T.copy()
    new_datapoints = np.ones((len(datapoints),1))
    new_datapoints = np.append(new_datapoints, datapoints, axis=1)
    new_datapoints = new_datapoints.T
    print(new_datapoints)
    VX = V @ new_datapoints
    print(VX)
    f = np.vectorize(heaviside_function)
    Y = f(VX, H_0)
    print(Y)
    new_Y = np.ones((len(Y.T), 1))
    new_Y = np.append(new_Y, Y.T, axis=1)
    new_Y = new_Y.T
    Z = weight @ new_Y
    print(f"The final response of the output neuron: {Z}")


# extreme_learning_machine(V, weight, datapoints, H_0=0.5)


def sparse_coding(V, x, y_t, LAMBDA = 0):
    error_result = np.zeros(len(y_t))
    for i in range(len(y_t)):
        error = np.subtract(x.T, (V.T @ y_t[i].T))
        error = np.linalg.norm(error) + LAMBDA * np.linalg.norm(y_t[i],ord=0)
        error_result[i]=error
        print(f"error for y_t{i+1}: {error}")
    print(f"best sparse code: y_t{np.argmin(error_result)+1}")

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

