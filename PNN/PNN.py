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
- KNN Classifier (Actually belongs to week2 but nevermind)
- Euclidean Distance (In KNN --> True)
- Manhattan Distance (In KNN --> False)
"""
# ==============================================================================

# w1.confusion_matrix(
#     y_pred = np.asarray([1, 0, 1, 1, 0, 1, 0], dtype = np.int8),
#     y_true = np.asarray([1, 1, 0, 1, 0, 1, 1], dtype = np.int8),
# )

# ==============================================================================
#
# w1.knn_classifier(
#     feature_vectors = np.asarray(
#         [
#             [5, 1],
#             [5, -1],
#             [3, 0],
#             [2, 1],
#             [4, 2]
#         ]
#     ),
#     given_target = np.asarray([4, 0]),
#     classes = np.asarray([1, 1, 2, 2, 2], dtype = np.int8),
#     k = 3,
#     euclidean_dist= True
# )

# =================================================================================
# euclidean_distance = w1.euclidean_distance(x = np.asarray([1, 2]), y= np.asarray([1, 1]))
# print(f"Euclidean distance is : {euclidean_distance}")

# manhattan_distance = w1.manhattan_distance(x = np.asarray([1.0, 2.0]), y= np.asarray([1.0, 1.0]))
# print(f"Manhattan distance is : {manhattan_distance}")
# =================================================================================

"""
Week 2 Discriminant Functions
- Classify class by given linear discriminant function (No learning process, just checking class)
- Dichotomizer determine class (with augmentation)
- Batch perceptron learning (with augmentation and sample normalisation)
- Sequential perceptron learning (with augmentation notation and sample normalisation)
- Sequential perceptron learning using_wk (with augmented notation and NO sample normalisation)
- Sequential multiclass learning 
- Sequential Widrowhoff
"""

# w2.classify_class_by_given_linear_discriminant_function(
#     initial_at= np.asarray([
#         [1, 0.5, 0.5],
#         [-1, 2, 2],
#         [2, -1, -1]
#     ]),
#     xt = np.asarray([
#         [0, 1],
#         [1, 0],
#         [0.5, 0.5],
#         [1, 1],
#         [0, 0]
#     ]),
#     true_label= np.asarray([1,1,2,2,3]) # keep empty np.asarray([]) if no label need to be check
# )
# ========================================================================
# w2.dichotomizer_determine_class(
#     initial_at = np.asarray([-1, 2, 2]),
#     xt = np.asarray(
#         [
#             [0, 1],
#         ]
#     )
# )
# ===========================================================================

# w2.batch_perceptron_learning(
#     initial_at = np.asarray([-25, 6, 3]),
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

# ==========================================================================
# w2.sequential_perceptron_learning_sample_normalisation(
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
# ============================================================================
# w2.sequential_perceptron_learning_using_wk(
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
# =============================================================================
# w2.sequential_multiclass_learning(
#     initial_at = np.asarray(
#         [
#             [1,0.5,0.5, -0.75],
#             [-1,2,2, 1],
#             [2,-1,-1, 1]
#         ]
#     ),
#     xt = np.asarray(
#         [
#             [0, 1, 0],
#             [1, 0, 0],
#             [0.5, 0.5, 0.25],
#             [1, 1, 1],
#             [0,0, 0]
#         ]
#     ),
#     true_label = np.asarray([1,1,2,2,3]),
#     learning_rate = 1.0,
#     epochs = 10
# )
# ===================================================================================
# w2.sequential_widrow_hoff(
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


"""
Week 3 

"""
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

"""
Week 6
 
"""



