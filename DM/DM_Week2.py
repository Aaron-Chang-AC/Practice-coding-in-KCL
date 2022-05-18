import numpy as np
from sympy import *

def sum_of_squared_error(y_hat, y):
    temp_error_list = []
    for i in range(len(y_hat)):
        error = np.linalg.norm(y_hat[i] - y[i])**2
        temp_error_list.append(error)
    final_error = sum(temp_error_list)
    print(f"The final sum of squared errors is: \n {final_error}")
    return final_error

def mean_squared_error(y_hat, y):
    temp_error_list = []
    for i in range(len(y_hat)):
        error = np.linalg.norm(y_hat[i] - y[i])**2
        temp_error_list.append(error)
    final_error = sum(temp_error_list)/len(temp_error_list)
    print(f"The final mean squared errors is: \n {final_error}")
    return final_error



