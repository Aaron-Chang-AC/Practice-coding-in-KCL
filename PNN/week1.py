import pandas as pd
from sklearn.datasets import load_iris
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import math
import collections


def confusion_matrix(y_pred,y_true):
    num = len(y_pred)
    TP=0.0
    FP=0.0
    TN=0.0
    FN=0.0
    # Calculate y & w_new
    for i in range(num):
        if y_pred[i]==1 and y_true[i]==1:
            TP+=1
        elif y_pred[i]==1 and y_true[i]==0:
            FP+=1
        elif y_pred[i]==0 and y_true[i]==0:
            TN+=1
        else:
            FN+=1
    print("TP:(pred = 1, true = 1)",int(TP))
    print("FP:(pred = 1, true = 0)",int(FP))
    print("TN:(pred = 0, true = 0)",int(TN))
    print("FN:(pred = 0, true = 1)",int(FN))
    print("error-rate:",(FP+FN)/(TP+FP+TN+FN))
    print("accuracy:",(TP+TN)/(TP+FP+TN+FN))
    print("recall:",TP/(TP+FN))
    print("precision:",TP/(TP+FP))
    print("f1-score",(2*TP)/(2*TP+FP+FN))

def euclidean_distance(x,y):
    temp=x-y
    return np.sqrt(np.dot(temp.T,temp))

def knn_classifier(feature_vectors,given_target,classes,k):
    num = len(feature_vectors)
    feature_vectors_class_map = {}
    feature_vectors_dist_map = {}
    for i in range(num):
        feature_vectors_class_map[i] = classes[i]
        feature_vectors_dist_map[i] = euclidean_distance(feature_vectors[i], given_target)
        print("x",str(i+1),": euclidean_distance=", str(round(feature_vectors_dist_map[i], 5)))

    distances_keys = sorted(feature_vectors_dist_map, key= feature_vectors_dist_map.get)
    sorted_distances_dict = {}
    for w in distances_keys:
        sorted_distances_dict[w] = feature_vectors_dist_map[w]
    print(sorted_distances_dict)

    neighbors=[]
    
    for i in range(k):
        neighbors.append(feature_vectors_class_map[list(sorted_distances_dict.keys())[i]])
    counter = collections.Counter(neighbors)
    print("The new sample is in class ",str(counter.most_common(1)[0][0]))

    return

#################EXECUTION

# confusion_matrix(
#     y_pred = np.asarray([1, 0, 1, 1, 0, 1, 0], dtype = np.int8),
#     y_true = np.asarray([1, 1, 0, 1, 0, 1, 1], dtype = np.int8),
# )
#
# knn_classifier(
#     feature_vectors = np.asarray(
#         [
#             [0.3, 0.35],
#             [0.3,0.28],
#             [0.24,0.2],
#             [0.2,0.32],
#             [0.12,0.25]
#         ]
#     ),
#     given_target = np.asarray([0.2, 0.25]),
#     classes = np.asarray([1, 2, 2, 3, 3], dtype = np.int8),
#     k = 3
# )
