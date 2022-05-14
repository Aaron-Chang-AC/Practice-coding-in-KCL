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


def find_weight_in_ffn(input, output, given_weight):
    # output of neural network z=(W_kj)*(W_ji)*x
    weights = np.zeros_like(given_weight)
    w_ji = np.linalg.inv(given_weight) @ output @ input.T @ np.linalg.inv((input @ input.T))
    weights = w_ji.copy()
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            print(f"w_{i+1}{j+1} = {weights[i][j]}")
    return weights

#
# input = np.array([
#     [2, -4],  # x1
#     [-0.5, -6]  # x2
# ])
#
# output = np.array([
#     [98, -168],  # z1
#     [7.5, -246]  # z2
# ])
#
# given_weight= np.array([
#     [8, -4],
#     [6, 9]
# ])
#
# print(find_weight_in_ffn(input, output, given_weight))
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