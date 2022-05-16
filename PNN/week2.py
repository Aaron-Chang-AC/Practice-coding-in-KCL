import pandas as pd
from sklearn.datasets import load_iris
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import math


def classify_class_by_given_linear_discriminant_function(initial_at, xt, true_label):
    at = initial_at
    num_at = len(at)
    num_xt = len(xt)

    # augmented notation with 1
    yt = np.ones((num_xt, 1), dtype=np.float32)
    yt = np.append(yt, xt, axis=1)

    feature_vector_length = len(yt[0])
    num_yt = len(yt)
    print(f"yt= \n {yt}")
    cnt = 0

    for i in range(num_yt):
        print(f"-----------------Iteration {cnt + 1}---------------")
        result_class = []
        for a in range(num_at):
            gt = np.matmul(at[a], yt[i].transpose()).transpose()
            print(f"gt{a+1}: {gt}")
            result_class.append(gt)
        if len(true_label) != 0:
            predicted_class = int(np.argmax(result_class)+1)
            true_class = int(true_label[i])
            if predicted_class != true_class:
                print(f"Misclassified, the original label is: class {true_class}")
        else:
            pass
        cnt += 1
        print(f"Maximum Response:{max(result_class)}")
    return



# Simple dichotomizer determines class of input with data augmentation
def dichotomizer_determine_class(initial_at,xt):
    """
    Simple dichotomizer determines class of input with data augmentation

    :param initial_at: [w_0, w_1, w_2]
    :param xt: Feature Vector
    :return: Class of feature vectors
    """
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
    """
    # (with augmented notation and sample normalisation)

    # a = (w0, wt)t

    # misclassified -> when g(x)<=0 !!

    :param initial_at:
    :param xt:
    :param true_label: the original class of xt
    :param learning_rate: alpha
    :param epochs:
    :return:
    """
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
        print("updated at=",at)
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
    """
    # (with augmented notation and sample normalisation)

    # a = (w0, wt)t

    # misclassified -> when g(x)<=0 !!

    :param initial_at:
    :param xt:
    :param true_label: True label (class) can only be 1 , 2....
    :param learning_rate:
    :param epochs:
    :return:
    """
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
    print(f"Final at: {at}")
    return

# (with augmented notation and no sample normalisation)
# a = (w0, wt)t
# misclassified -> when g(x)<=0 !!
def sequential_perceptron_learning_using_wk(initial_at,xt,true_label,learning_rate,epochs):
    """
    # (with augmented notation and no sample normalisation)

    # a = (w0, wt)t

    # misclassified -> when g(x)<=0 !!

    :param initial_at:
    :param xt:
    :param true_label:
    :param learning_rate:
    :param epochs:
    :return:
    """
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
            print(f"Final aT is: {at}")
            break
    return

# If more than one discriminant function produces the maximum output,
# choose the function with the highest index (i.e., the one that represents
# the largest class label)!!
def sequential_multiclass_learning(initial_at,xt,true_label,learning_rate,epochs,select_highest_index=True):
    """
    # If more than one discriminant function produces the maximum output,

    # choose the function with the highest index (i.e., the one that represents

    # the largest class label)!!

    :param initial_at: Number of inital_at depends on the number of class (true label)
    :param xt:
    :param true_label:
    :param learning_rate:
    :param epochs:
    :return:
    """
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
    cnt = 0
    for e in range(epochs):
        print("\n\n=====epoch ",str(e+1),"=======")
        for i in range(num_yt):
            print(f"-----------------Iteration {cnt+1}---------------")
            gt = np.matmul(at, yt[i].transpose()).transpose()
            print("gt:",gt)


            reversed_gt = gt[::-1]
            result_class = int(len(reversed_gt) - np.argmax(reversed_gt))

            # if we need to select the lowest index
            if not(select_highest_index):
                reversed_gt = gt.copy()
                result_class = np.argmax(reversed_gt) + 1

            true_class = int(true_label[i])
            if result_class != true_class:
                at[true_class-1] = at[true_class-1] + learning_rate * yt[i]
                at[result_class-1] = at[result_class-1] - learning_rate * yt[i]
                misclassified[i] = 1
                print("y", str(i), ":gt=", gt, "pred_class: ", str(result_class) ," true_class: ", str(true_class), " misclassified")
                print(f"updated at= \n {at}")
            else:
                print("y", str(i), ":gt=", gt, "pred_class: ", str(result_class) ," true_class: ", str(true_class), " correct")
            cnt += 1
        print("\n\n")
        if np.sum(misclassified, dtype=np.int8) > 0:
            misclassified = np.zeros(num_xt, dtype=np.int8)
        else:
            break

    print(f"Final result at (each ROW is a linear discriminant function):\n {at}")
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
#     epochs = 10,
#     select_highest_index=False
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