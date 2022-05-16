import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
import math

# output y = H(wx)
def heaviside_function(H_0, wx):
    result = 0.0
    if wx < (10 ** (-9)) and wx > (10 ** (-9) * (-1)):
        result = H_0
    elif wx > 0:
        result = 1
    else:
        result = 0

    return result

# With augmented notations!!
# Note: theta is equivelent to -w0
def simple_neuron(initial_theta,initial_w,xt,H_0):
    w0 = -initial_theta
    w = np.append(np.asarray([w0], dtype=np.float32), initial_w, axis=0)
    num_xt = len(xt)

    # augmented notation with 1
    xt_new = np.ones((num_xt, 1), dtype=np.float32)
    xt_new = np.append(xt_new, xt, axis=1)

    for i in range(num_xt):
        result = np.matmul(w,xt_new[i].transpose())
        print("y",str(i+1),"=H(",str(round(result,5)),")=",str(round(heaviside_function(H_0, result),5)))
    
    return


# With augmented notations!!
# Note: theta is equivelent to -w0
def sequential_delta_learning_rule(initial_theta, initial_w, xt, true_values, H_0, learning_rate, epochs):
    w0 = -initial_theta
    w = np.append(np.asarray([w0], dtype=np.float32), initial_w, axis=0)
    num_xt = len(xt)

    # augmented notation with 1
    xt_new = np.ones((num_xt, 1), dtype=np.float32)
    xt_new = np.append(xt_new, xt, axis=1)

    t = true_values

    for e in range(epochs):
        print(f"-------Epoch {e+1}-------")
        for i in range(num_xt):
            wx = np.matmul(w, xt_new[i].transpose())
            y = heaviside_function(H_0, wx)
            delta_w = learning_rate * (t[i]-y) * xt_new[i]
            w = w + delta_w
            print("x", str(i + 1), "=", xt_new[i], " , t=", str(t[i]), " , y =H(", str(round(wx, 5)) , ")=",str(y), " , (t−y)=",str(t[i]-y), " , η*(t-y)*xt=",delta_w, " , w_new=",w)
            print("\n")
    return

def batch_delta_learning_rule(initial_theta, initial_w, xt, true_values, H_0, learning_rate, epochs):
    w0 = -initial_theta
    w = np.append(np.asarray([w0], dtype=np.float32), initial_w, axis=0)
    len_w = len(w)
    num_xt = len(xt)

    # augmented notation with 1
    xt_new = np.ones((num_xt, 1), dtype=np.float32)
    xt_new = np.append(xt_new, xt, axis=1)

    t = true_values

    for e in range(epochs):
        total_delta_w=np.zeros(len_w, dtype=np.float32)
        print("\n\nepoch ", str(e + 1), ":")
        for i in range(num_xt):
            wx = np.matmul(w, xt_new[i].transpose())
            y = heaviside_function(H_0, wx)
            delta_w = learning_rate * (t[i]-y) * xt_new[i]
            total_delta_w = total_delta_w + delta_w
            print("x", str(i + 1), "=", xt_new[i], " , t=", str(t[i]), " , y =H(", str(round(wx, 5)) , ")=",str(y), " , (t−y)=",str(t[i]-y), " , η*(t-y)*xt=",delta_w)
        w = w + total_delta_w
        print("total_delta_w= ",total_delta_w," , w_new= ",w)

    print("Remember theta has a negative, answer given below")
    print(f"theta = -{w[0]}, the rest of weight w1, w2,..... = {w[1:]}")
    return

# e = x − WT * y
# y ← y + alpha * W *e
# no augmented notation
def negative_feedback_network_original(initial_W, initial_yt , xt , alpha, epochs):
    W = initial_W
    yt = initial_yt
    num_xt = len(xt)

    for e in range(epochs):
        print("\n\nepoch ", str(e + 1), ":")
        for i in range(num_xt):
            et = xt[i] - np.matmul(yt, W)
            We_t=np.matmul(et, W.transpose())
            yt = yt + alpha * We_t
            Wty_t = np.matmul(yt, W)
            print("et: ", et, " , (We)t: ", We_t, " , yt: ", yt, " , (Wt*y)t:", Wty_t)

    print("\n\nOutput is(if 1d array, y will be flattened):",yt.transpose())
    return

def negative_feedback_network_stable(initial_W, initial_yt, xt, alpha, epochs, epsilon1, epsilon2):
    W = initial_W
    yt = initial_yt
    num_xt = len(xt)

    sum_of_rows = W.sum(axis=1)
    W_dash = W / sum_of_rows[:, np.newaxis]
    print(W_dash)
    for e in range(epochs):
        print("\n\nepoch ", str(e + 1), ":")
        for i in range(num_xt):
            # decision between maximum epsilon or W_Ty
            WTy = np.maximum(epsilon2, np.matmul(yt, W))
            et = xt[i] / WTy
            Wdash_eT = np.matmul(et, W_dash.transpose())
            yt = np.maximum(epsilon1, yt)
            yt =  yt  * Wdash_eT
            Wty_t = np.matmul(yt, W)
            print("et: ", et, " , Wdash_eT: ", Wdash_eT, " , yt: ", yt, " , (Wt*y)t:", Wty_t)

    print("\n\nOutput is(if 1d array, y will be flattened):",yt.transpose())



# negative_feedback_network_stable(
#     initial_W = np.asarray(
#         [
#             [1,1,0],
#             [1,1,1]
#         ],
#         dtype=np.float32
#     ),
#     initial_yt = np.asarray([0, 0], dtype=np.float32),
#     xt=np.asarray(
#         [
#             [1,1,0],
#         ],
#         dtype=np.float32
#     ),
#     alpha=0.25,
#     epochs=5,
#     epsilon1= 0.01,
#     epsilon2= 0.01
#
# )

##########################################EXECUTION
# simple_neuron(
#     initial_theta = 0.0,
#     initial_w = np.asarray([0.1,-0.5,0.4], dtype=np.float32),
#     xt = np.asarray(
#         [
#             [0.1, -0.5, 0.4],
#             [0.1, 0.5, 0.4]
#         ]
#     ),
#     H_0 = 0.5
#
# )
#
# sequential_delta_learning_rule(
#     initial_theta=-1.0,
#     initial_w = np.asarray([0,0], dtype=np.float32),
#     xt=np.asarray(
#         [
#             [0,2],
#             [1,2],
#             [2,1],
#             [-3,1],
#             [-2,-1],
#             [-3,-2],
#         ],
#         dtype=np.float32
#     ),
#     true_values = np.asarray([1,1,1,0,0,0], dtype=np.float32),
#     H_0=0.5,
#     learning_rate=1.0,
#     epochs=10
#
# )
#
# batch_delta_learning_rule(
#     initial_theta=1.5,
#     initial_w = np.asarray([2], dtype=np.float32),
#     xt=np.asarray(
#         [
#             [0],
#             [1]
#         ],
#         dtype=np.float32
#     ),
#     true_values = np.asarray([1, 0], dtype=np.float32),
#     H_0=0.5,
#     learning_rate=1.0,
#     epochs=7
#
# )
#
# #NOTE: the dimension of yt is same as
# # (W * xt)t
# negative_feedback_network_original(
#     initial_W = np.asarray(
#         [
#             [1,1,0],
#             [1,1,1]
#         ],
#         dtype=np.float32
#     ),
#     initial_yt = np.asarray([0, 0], dtype=np.float32),
#     xt=np.asarray(
#         [
#             [1,1,0],
#         ],
#         dtype=np.float32
#     ),
#     alpha=0.5,
#     epochs=5
#
# )
