import week1 as w1
import week2 as w2
import week3 as w3
import week4 as w4
import week5 as w5
import week6 as w6
import week7 as w7
import week8 as w8
import week9 as w9
import week10 as w10
import numpy as np


"""
Week 1 Introduction
- Confusion Matrix
- Euclidean Distance
- Manhatten Distance 
- KNN Classifier
"""
# w1.confusion_matrix(
#     y_pred = np.asarray([1, 0, 1, 1, 0, 1, 0], dtype = np.int8),
#     y_true = np.asarray([1, 1, 0, 1, 0, 1, 1], dtype = np.int8),
# )

w1.knn_classifier(
    feature_vectors = np.asarray(
        [
            [0.3, 0.35],
            [0.3,0.28],
            [0.24,0.2],
            [0.2,0.32],
            [0.12,0.25]
        ]
    ),
    given_target = np.asarray([0.2, 0.25]),
    classes = np.asarray([1, 2, 2, 3, 3], dtype = np.int8),
    k = 3,
    euclidean_dist= False
)

"""
Week 2 Discriminant Functions
- Dichotomizer determine class
- Batch perceptron learning
- Sequential perceptron learning (sample normalisation)
- Sequential multiclass learning
- Sequential Widrowhoff
"""



"""
Week 3

"""


"""
Week 6
 
"""



