import numpy as np
import math
import numpy as np
from sympy import *

class activation_function_spec:
    def __init__(self, alpha=None, H_0=None, threshold=None, name=None):
        self.alpha = alpha
        self.H_0 = H_0
        self.threshold = threshold
        self.name = name

def rbf_function(dist,sigma,beta = 0.5,function = "gaussian"):
    if function == "gaussian":
        return math.exp(-0.5 * (dist**2) / (sigma**2))
    elif function == "multi_quadric":
        return ((dist ** 2) + (sigma ** 2)) ** 0.5
    elif function == "generalized_multi_quadric":
        return ((dist ** 2) + (sigma ** 2)) ** beta
    elif function == "inverse_multi_quadric":
        return ((dist ** 2) + (sigma ** 2)) ** (-0.5)
    elif function == "generalized_inverse_multi_quadric":
        return ((dist ** 2) + (sigma ** 2)) ** (-beta)
    elif function == "thin_plate_spline":
        if math.isclose(dist,0.0):
            return -np.Inf
        else:
            return (dist ** 2) * math.log(dist)
    elif function == "cubic":
        return dist ** 3
    elif function == "square":
        return dist ** 2
    elif function == "linear":
        return dist
    else:
        return 0.0
def rbf_network(X,true_label,centers,rho=None,samples_to_be_classified=[],beta = 0.5,function = "gaussian", rho_j_method = "max"):
    n = len(X)
    n_new = len(samples_to_be_classified)
    num_centers = len(centers)
    rho_j_list = []
    rho_j = 0.0

    for i in range(num_centers):
        for j in range(num_centers):
            if i != j:
                rho_j_list.append(np.sqrt(np.sum((centers[i]-centers[j])**2)))
    rho_j_list = np.asarray(rho_j_list)

    if rho_j_method == "max":
        rho_j = np.max(rho_j_list) / ((2.0*len(centers[0]))**0.5)
    else:
        rho_j = 2 * np.mean(rho_j_list) # 2 * mean

    if rho is not None:
        rho_j=rho

    print(f"rho_j: {rho_j}\n")
    y_list = []
    y_list_new = []
    for i in range(n):
        y_list.append([])
        for j in range(num_centers):
            temp = np.sqrt(np.sum((X[i]-centers[j])**2))
            y_list[i].append(rbf_function(temp,rho_j,beta,function))
        y_list[i].append(1.0)
    y_list=np.asarray(y_list)
    print(f"y_list (each row is the hidden output of a sample):\n{np.round(y_list,5)}\n")
    weights = np.linalg.pinv(y_list) @ true_label
    print(f"weight matrix:\n{np.round(weights,5)}")
    result = y_list @ weights
    print(f"--------classify original samples--------")
    for i in range(n):
        print(f"output of sample{i+1}: {result[i]}")

    print(f"--------classify new samples--------")
    for i in range(n_new):
        y_list_new.append([])
        for j in range(num_centers):
            temp = np.sqrt(np.sum((samples_to_be_classified[i]-centers[j])**2))
            y_list_new[i].append(rbf_function(temp,rho_j,beta,function))
        y_list_new[i].append(1.0)
    y_list_new=np.asarray(y_list_new)
    print(f"y_list_new (each row is the hidden output of a sample):\n{np.round(y_list_new,5)}\n")
    result_new = y_list_new @ weights
    for i in range(n_new):
        print(f"output of new sample{i+1}: {result_new[i]}")
    return

def sgn(input):
    if (input < (10 ** (-9))) and (input > (10 ** (-9) * (-1))):
        return 1.0
    elif input > 0.0:
        return 1.0
    else:
        return -1.0

def sigmoid(input):
  return 1 / (1 + math.exp(-input))

def heaviside(input, H_0, threshold):
    result = 0.0
    x = round(input - threshold, 5 )
    if (x < (10 ** (-9))) and (x > (10 ** (-9) * (-1))):
        result = H_0
    elif x > 0.0:
        result = 1.0
    else:
        result = 0.0

    return result

def relu(input):
    if (input < (10 ** (-9))) and (input > (10 ** (-9) * (-1))):
        return input
    elif input > 0.0:
        return input
    else:
        return 0.0

def lrelu(input,alpha):
    if (input < (10 ** (-9))) and (input > (10 ** (-9) * (-1))):
        return input
    elif input > 0.0:
        return input
    else:
        return alpha * input

def activation_function(input, alpha=None, H_0=None, threshold=None, mode=None):
    heaviside_f = np.vectorize(heaviside)
    relu_f = np.vectorize(relu)
    lrelu_f = np.vectorize(lrelu)
    sigmoid_f = np.vectorize(sigmoid)
    sgn_f = np.vectorize(sgn)
    if mode == "relu":
        return relu_f(input)
    elif mode == 'lrelu':
        return lrelu_f(input,alpha)
    elif mode == 'tanh':
        return np.tanh(input)
    elif mode == "heaviside":
        return heaviside_f(input, H_0, threshold)
    elif mode == 'linear':
        return input
    elif mode == 'sigmoid':
        return sigmoid_f(input)
    elif mode == 'sgn':
        return sgn_f(input)

def neural_network_forward(weight_dict=None, input=None, activation_function_dict=None):
    np.set_printoptions(formatter={'float_kind': '{:f}'.format})
    # calculate the result of the input layer
    X = input.copy()
    input_spec = activation_function_dict['input']
    X = activation_function(X, alpha=input_spec.alpha, H_0=input_spec.H_0, threshold=input_spec.threshold, mode=input_spec.name)
    num_all_layers = len(activation_function_dict)
    print(f"result of the input layer (each row is a sample):\n{X}")

    # calculate the result of hidden layers
    for i in range(num_all_layers - 2):
        X = (weight_dict['W'+str(i+1)] @ X.T + weight_dict['B'+str(i+1)].T).T
        temp_spec = activation_function_dict['hidden'+str(i+1)]
        X = activation_function(X, alpha=temp_spec.alpha, H_0=temp_spec.H_0, threshold=temp_spec.threshold,
                                    mode=temp_spec.name)
        print(f"result of hidden layer{i+1} (each row is a sample):\n{X}")

    # calculate the result of the output layer
    X = (weight_dict['W' + str(num_all_layers - 1)] @ X.T + weight_dict['B' + str(num_all_layers - 1)].T).T
    output_spec = activation_function_dict['output']
    X = activation_function(X, alpha=output_spec.alpha, H_0=output_spec.H_0, threshold=output_spec.threshold,
                                mode=output_spec.name)
    print(f"result of the output layer (each row is a sample):\n{X}")

    return X

def inverted_array(phi, t):
    return np.inner(np.linalg.pinv(phi), t)

def func_diff(function = None, value = None):
    x = symbols('x', real=True)
    f_d = diff(function, x)
    return f_d.subs(x,value)
'''
tutorial Q4: forward NN
X = np.asarray([
    [1,0,1,0],
    [0,1,0,1],
    [1,1,0,0]
],dtype=np.float32)

W = {
    'W1': np.asarray([
        [-0.7057,1.9061,2.6605,-1.1359],
        [0.4900,1.9324,-0.4269,-5.1570],
        [0.9438,-5.4160,-0.3431,-0.2931]
    ]),
    'B1': np.asarray([
        [4.8432,0.3973,2.1761]
    ]),
    'W2': np.asarray([
        [-1.1444,0.3115,-9.9812],
        [0.0106,11.5477,2.6479]
    ]),
    'B2': np.asarray([
        [2.5230,2.6463]
    ]),
}
activation_function_dict = {
    'input': activation_function_spec(name='linear'),
    'hidden1': activation_function_spec(name='tanh'),
    'output': activation_function_spec(name='sigmoid'),
}
neural_network_forward(weight_dict=W, input=X, activation_function_dict=activation_function_dict)
'''

'''

phi = np.array([
    [1, 0.1353, 1],
    [0.3679, 0.3679, 1],
    [0.3679, 0.3679, 1],
    [0.1353, 1, 1]
])
t = np.array([0, 1, 1, 0])
w = inverted_array(phi, t)
print(w)
print(f"Dot back the result:{np.round(np.inner(phi, w),3)}")
'''

'''
for func_diff
x = symbols('x', real=True)
func = 2/(1+exp(-(2*x)))-1
print(func.subs(x,3.0))
print(0.3-0.25*(-0.7869-0.5)*func_diff(function = func,value = -1.0633)*1.6*func_diff(function = func,value = -0.6)*0.1)
'''

# for rbf_network version 1
X = np.asarray([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
],dtype=np.float32)

true_label = np.asarray([
    [0],
    [1],
    [1],
    [0]
],dtype=np.float32)

centers = np.asarray([
    [0,0],
    [1,1]
],dtype=np.float32)

samples_to_be_classified = np.asarray([
    [0.5,-0.1],
    [-0.2,1.2],
    [0.8,0.3],
    [1.8,0.6]
],dtype=np.float32)
rbf_network(X = X,
            true_label = true_label,
            centers = centers,
            rho = None,
            samples_to_be_classified = samples_to_be_classified,
            beta = 0.5,
            function = "gaussian",
            rho_j_method = "max"
            )

# for rbf_network version 2
# X = np.asarray([
#     [0.05],
#     [0.2],
#     [0.25],
#     [0.3],
#     [0.4],
#     [0.43],
#     [0.48],
#     [0.6],
#     [0.7],
#     [0.8],
#     [0.9],
#     [0.95]
# ],dtype=np.float32)
# true_label = np.asarray([
#     [0.0863],
#     [0.2662],
#     [0.2362],
#     [0.1687],
#     [0.126],
#     [0.1756],
#     [0.3290],
#     [0.6694],
#     [0.4573],
#     [0.332],
#     [0.4063],
#     [0.3535]
# ],dtype=np.float32)
#
# centers = np.asarray([
#     [0.2],
#     [0.6],
#     [0.9]
# ],dtype=np.float32)
#
# samples_to_be_classified = np.asarray([
#     [0.1],
#     [0.35],
#     [0.55],
#     [0.75],
#     [0.9]
# ],dtype=np.float32)
# rbf_network(X = X,
#             true_label = true_label,
#             centers = centers,
#             rho = 0.1,
#             samples_to_be_classified = samples_to_be_classified,
#             beta = 0.5,
#             function = "gaussian",
#             rho_j_method = "max"
#             )

# for rbf_network version 3
# X = np.asarray([
#     [0.05],
#     [0.2],
#     [0.25],
#     [0.3],
#     [0.4],
#     [0.43],
#     [0.48],
#     [0.6],
#     [0.7],
#     [0.8],
#     [0.9],
#     [0.95]
# ],dtype=np.float32)
# true_label = np.asarray([
#     [0.0863],
#     [0.2662],
#     [0.2362],
#     [0.1687],
#     [0.126],
#     [0.1756],
#     [0.3290],
#     [0.6694],
#     [0.4573],
#     [0.332],
#     [0.4063],
#     [0.3535]
# ],dtype=np.float32)
#
# centers = np.asarray([
#     [0.1667],
#     [0.35],
#     [0.5525],
#     [0.8833]
# ],dtype=np.float32)
#
# samples_to_be_classified = np.asarray([
#     [0.1],
#     [0.35],
#     [0.55],
#     [0.75],
#     [0.9]
# ],dtype=np.float32)
# rbf_network(X = X,
#             true_label = true_label,
#             centers = centers,
#             rho = None,
#             samples_to_be_classified = samples_to_be_classified,
#             beta = 0.5,
#             function = "thin_plate_spline",
#             rho_j_method = "average"
#             )
