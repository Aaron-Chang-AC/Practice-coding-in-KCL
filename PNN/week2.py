import pandas as pd
from sklearn.datasets import load_iris
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import math

# Simple dichotomizer determines class of input with data augmentation
def dichotomizer_determine_class(initial_at,xt):
    at=initial_at
    num_xt=len(xt)
    yt=np.ones((num_xt,1), dtype=np.float32)

    yt=np.append(yt,xt,axis=1)
    feature_vector_length = len(yt[0])
    print("yt=")
    print(yt)

    for i in range(num_xt):
        result = np.matmul(at,yt[i].transpose())
        if result < (10 ** (-9)) and result > (10 ** (-9) * (-1)):
            print("x", str(i), ": g(x)=", str(round(result,5)), " class ", "UNKNOWN")
        elif result > 0:
            print("x", str(i), ": g(x)=", str(round(result, 5)), " class 1")
        else:
            print("x", str(i), ": g(x)=", str(round(result, 5)), " class 2")
    return

# (with augmented notation and sample normalisation)
# a = (w0, wt)t
# misclassified -> when g(x)<=0 !!
def batch_perceptron_learning(initial_at,xt,true_label,learning_rate,epochs):
    at = initial_at
    num_xt = len(xt)

    # augmented notation with 1
    yt = np.ones((num_xt, 1), dtype=np.float32)
    yt = np.append(yt, xt, axis=1)

    # sample normalisation
    for i in range(num_xt):
        if true_label[i] > 1:
            yt[i] = yt[i]*(-1)

    feature_vector_length = len(yt[0])
    print("yt=")
    print(yt)

    misclassified=np.zeros(num_xt, dtype=np.int8)
    for e in range(epochs):
        print("\n\nepoch ",str(e+1),":")
        temp = np.zeros(feature_vector_length, dtype=np.float32)
        for i in range(num_xt):
            result = np.matmul(at, yt[i].transpose())
            if result < (10 ** (-9)) and result > (10 ** (-9) * (-1)):
                temp+=yt[i]
                misclassified[i]=1
                print("y", str(i), ":g(x)=", str(round(result, 5)), " misclassified")
            elif result > 0:
                print("y", str(i), ":g(x)=", str(round(result, 5)), " correct")
                continue
            else:
                temp+=yt[i]
                misclassified[i] = 1
                print("y", str(i), ":g(x)=", str(round(result, 5)), " misclassified")

        # update at
        at = at + learning_rate*temp
        print("at=",at)
        print("\n\n")
        if np.sum(misclassified, dtype=np.int8) > 0:
            misclassified = np.zeros(num_xt, dtype=np.int8)
        else:
            break
    return

# (with augmented notation and sample normalisation)
# a = (w0, wt)t
# misclassified -> when g(x)<=0 !!
def sequential_perceptron_learning_sample_normalisation(initial_at,xt,true_label,learning_rate,epochs):
    at = initial_at
    num_xt = len(xt)

    # augmented notation with 1
    yt = np.ones((num_xt, 1), dtype=np.float32)
    yt = np.append(yt, xt, axis=1)

    # sample normalisation
    for i in range(num_xt):
        if true_label[i] > 1:
            yt[i] = yt[i]*(-1)

    feature_vector_length = len(yt[0])
    print("yt=")
    print(yt)

    misclassified=np.zeros(num_xt, dtype=np.int8)
    for e in range(epochs):
        print("\n\nepoch ",str(e+1),":")
        for i in range(num_xt):
            result = np.matmul(at, yt[i].transpose())
            if result < (10 ** (-9)) and result > (10 ** (-9) * (-1)):
                at = at + learning_rate * yt[i]
                misclassified[i]=1
                print("y", str(i), ":g(x)=", str(round(result, 5)), " misclassified")
            elif result > 0:
                print("y", str(i), ":g(x)=", str(round(result, 5)), " correct")
                continue
            else:
                at = at + learning_rate * yt[i]
                misclassified[i] = 1
                print("y", str(i), ":g(x)=", str(round(result, 5)), " misclassified")

            # update at
            print("at=",at)

        print("\n\n")
        if np.sum(misclassified, dtype=np.int8) > 0:
            misclassified = np.zeros(num_xt, dtype=np.int8)
        else:
            break
    return

# (with augmented notation and no sample normalisation)
# a = (w0, wt)t
# misclassified -> when g(x)<=0 !!
def sequential_perceptron_learning_using_wk(initial_at,xt,true_label,learning_rate,epochs):
    at = initial_at
    num_xt = len(xt)

    # augmented notation with 1
    yt = np.ones((num_xt, 1), dtype=np.float32)
    yt = np.append(yt, xt, axis=1)

    # inverse class 2 labels
    true_label = np.where(true_label>1,-1,true_label)
    print("true labels:",true_label)
    feature_vector_length = len(yt[0])
    print("yt=")
    print(yt)

    misclassified=np.zeros(num_xt, dtype=np.int8)
    for e in range(epochs):
        print("\n\nepoch ",str(e+1),":")
        for i in range(num_xt):
            result = np.matmul(at, yt[i].transpose())
            if result < (10 ** (-9)) and result > (10 ** (-9) * (-1)):
                at = at + learning_rate * true_label[i] * yt[i]
                misclassified[i]=1
                print("y", str(i), ":g(x)=", str(round(result, 5)), " misclassified")
            elif result > 0:
                if true_label[i] > 0:
                    print("y", str(i), ":g(x)=", str(round(result, 5)), " correct")
                    continue
                else:
                    at = at + learning_rate * true_label[i] * yt[i]
                    misclassified[i] = 1
                    print("y", str(i), ":g(x)=", str(round(result, 5)), " misclassified")
            else:
                if true_label[i] > 0:
                    at = at + learning_rate * true_label[i] * yt[i]
                    misclassified[i] = 1
                    print("y", str(i), ":g(x)=", str(round(result, 5)), " misclassified")
                else:
                    print("y", str(i), ":g(x)=", str(round(result, 5)), " correct")
                    continue
            # update at
            print("at=",at)

        print("\n\n")
        if np.sum(misclassified, dtype=np.int8) > 0:
            misclassified = np.zeros(num_xt, dtype=np.int8)
        else:
            break
    return

# If more than one discriminant function produces the maximum output,
# choose the function with the highest index (i.e., the one that represents
# the largest class label)!!
def sequential_multiclass_learning(initial_at,xt,true_label,learning_rate,epochs):
    at = initial_at
    num_at = len(at)
    num_xt = len(xt)
    
    # augmented notation with 1
    yt = np.ones((num_xt, 1), dtype=np.float32)
    yt = np.append(yt, xt, axis=1)

    feature_vector_length = len(yt[0])
    num_yt =  len(yt)
    print("yt=")
    print(yt)

    misclassified=np.zeros(num_yt, dtype=np.int8)

    for e in range(epochs):
        print("\n\nepoch ",str(e+1),":")
        for i in range(num_yt):
            gt = np.matmul(at, yt[i].transpose()).transpose()
            print("gt:",gt)
            reversed_gt = gt[::-1]
            result_class = int(len(reversed_gt) - np.argmax(reversed_gt))

            true_class = int(true_label[i])
            if result_class != true_class:
                at[true_class-1] = at[true_class-1] + learning_rate * yt[i]
                at[result_class-1] = at[result_class-1] - learning_rate * yt[i]
                misclassified[i] = 1
                print("y", str(i), ":gt=", gt, "pred_class: ", str(result_class) ," true_class: ", str(true_class), " misclassified")
                print("at=",at)
            else:
                print("y", str(i), ":gt=", gt, "pred_class: ", str(result_class) ," true_class: ", str(true_class), " correct")
        print("\n\n")
        if np.sum(misclassified, dtype=np.int8) > 0:
            misclassified = np.zeros(num_xt, dtype=np.int8)
        else:
            break
    return

def sequential_widrow_hoff(initial_at,xt,true_label,margin_vector,learning_rate,epochs):
    at = initial_at
    num_xt = len(xt)

    # augmented notation with 1
    yt = np.ones((num_xt, 1), dtype=np.float32)
    yt = np.append(yt, xt, axis=1)

    # sample normalisation
    for i in range(num_xt):
        if true_label[i] > 1:
            yt[i] = yt[i] * (-1)

    feature_vector_length = len(yt[0])
    print("yt=")
    print(yt)

    iteraton=0
    for e in range(epochs):
        print("\n\nepoch ", str(e + 1), ":")
        for i in range(num_xt):
            iteraton+=1
            result = np.matmul(at, yt[i].transpose())

            # update at
            at = at + learning_rate * (margin_vector[i] - result) * yt[i]
            print("Iteration ",str(iteraton),": at*yk = ",str(round(result, 5)), " at_new = ", at)
    return

##########################################EXECUTION
# dichotomizer_determine_class(
#     initial_at = np.asarray([-3,1,2,2,2,4]),
#     xt = np.asarray(
#         [
#             [0,-1,0,0,1],
#             [1,1,1,1,1]
#         ]
#     )
# )
#
# batch_perceptron_learning(
#     initial_at = np.asarray([-25,6,3]),
#     xt = np.asarray(
#         [
#             [1,5],
#             [2,5],
#             [4,1],
#             [5,1]
#         ]
#     ),
#     true_label = np.asarray([1,1,2,2]),
#     learning_rate = 1.0,
#     epochs = 10
# )
#
# sequential_perceptron_learning_sample_normalisation(
#     initial_at = np.asarray([1,0,0]),
#     xt = np.asarray(
#         [
#             [0,2],
#             [1,2],
#             [2,1],
#             [-3,1],
#             [-2,-1],
#             [-3,-2]
#         ]
#     ),
#     true_label = np.asarray([1,1,1,2,2,2]),
#     learning_rate = 1.0,
#     epochs = 10
# )
#
# sequential_perceptron_learning_using_wk(
#     initial_at = np.asarray([1,0,0]),
#     xt = np.asarray(
#         [
#             [0,2],
#             [1,2],
#             [2,1],
#             [-3,1],
#             [-2,-1],
#             [-3,-2]
#         ]
#     ),
#     true_label = np.asarray([1,1,1,2,2,2]),
#     learning_rate = 1.0,
#     epochs = 10
# )
#
# sequential_multiclass_learning(
#     initial_at = np.asarray(
#         [
#             [0,0,0],
#             [0,0,0],
#             [0,0,0]
#         ]
#     ),
#     xt = np.asarray(
#         [
#             [1,1],
#             [2,0],
#             [0,2],
#             [-1,1],
#             [-1,-1]
#         ]
#     ),
#     true_label = np.asarray([1,1,2,2,3]),
#     learning_rate = 1.0,
#     epochs = 10
# )
#
# sequential_widrow_hoff(
#     initial_at = np.asarray([1,0,0]),
#     xt=np.asarray(
#         [
#             [0, 2],
#             [1, 2],
#             [2, 1],
#             [-3, 1],
#             [-2, -1],
#             [-3, -2]
#         ]
#     ),
#     true_label = np.asarray([1,1,1,2,2,2]),
#     margin_vector = np.asarray([1,1,1,1,1,1]).transpose(),
#     learning_rate = 0.1,
#     epochs = 2
# )