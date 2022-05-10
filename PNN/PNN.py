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
from sympy import *


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
#     epochs = 10,
#     select_highest_index=True
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


# ======================================================

# w3.simple_neuron(
#     initial_theta = -2,
#     initial_w = np.asarray([-1, 3], dtype=np.float32),
#     xt = np.asarray(
#         [
#             [2, 0.5]
#         ]
#     ),
#     H_0 = 1
#
# )
# ============================================================
# w3.sequential_delta_learning_rule(
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

# =============================================================
# w3.batch_delta_learning_rule(
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
# ================================================================
# #NOTE: the dimension of yt is same as
# # (W * xt)t
# w3.negative_feedback_network_original(
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
#     epochs=4
#
# )

"""
Week 4

"""


"""
Week 5
- Activation Function (ReLU, LReLU, tanh, heavside)
- Batch Normalization
- Mask Convolution (Refer to code if possible, bit complicated)

# For pooling----
- Average Pooling
- Max Pooling
"""
# ====================================================================================================
# # Activation_function

# input = np.asarray([
#     [1, 0.5, 0.2],
#     [-1, -0.5, -0.2],
#     [0.1, -0.1, 0]
# ], dtype=np.float32)
# alpha = 0.1
# threshold = 0.1
# H_0 = 0.5
#
# print(f'ReLU result: \n {w5.activation_function(input, mode="relu")}')
# print(f'LReLU result: \n {w5.activation_function(input, alpha=0.1, mode="lrelu")}')
# print(f'Tanh result: \n {w5.activation_function(input, mode="tanh")}')
# print(f'Heaviside result: \n {w5.activation_function(input, H_0=0.5, threshold=0.1, mode="heaviside")}')
# ====================================================================================================

# # Batch normalization for output neurons
# input1 = np.array([
#     [1, 0.5,  0.2],
#     [ -1, -0.5,  -0.2],
#     [0.1, -0.1,  0],
# ])
#
# input2 = np.array([
#     [1, -1,  0.1],
#     [0.5, -0.5,  -0.1],
#     [0.2, -0.2,  0],
# ])
#
# input3 = np.array([
#     [0.5, -0.5,  -0.1],
#     [ 0, -0.4,  0],
#     [0.5, 0.5,  0.2],
# ])
#
# input4 = np.array([
#     [0.2, 1,  -0.2],
#     [-1, -0.6,  -0.1],
#     [0.1, 0,  0.1],
# ])
#
# final_input_array = np.stack([input1, input2, input3, input4])
# # print(final_input_array)
# print(w5.batch_normalization(final_input_array, beta=0, gamma=1, eta=0.1))

# ====================================================================================================
# For the purpose of image convolution with mask H
# multiple channels and each channel corresponds to a mask
# padding=0, stride=1, dilation=2:

# conv_input1 = np.array([
#     [0.2, 1,  0],
#     [ -1, 0,  -0.1],
#     [0.1, 0,  0.1]
# ])
# conv_input2 = np.array([
#     [1, 0.5,  0.2],
#     [ -1, -0.5,  -0.2],
#     [0.1, -0.1,  0]
# ])
# H1 = np.array([
#     [1,  -0.1],
#     [1,  -0.1]
# ])
# H2 = np.array([
#     [0.5,  0.5],
#     [-0.5, -0.5]
# ])
# # parameters setting
# padding=0
# stride=1
# dilation = 2
# use_dilation = True # set True if dilation is used
#
# if use_dilation:
#     # only use when dilation is involved
#     H1_after_dilation1 = w5.mask_dilation(H1, dilation=2)
#     H2_after_dilation2 = w5.mask_dilation(H2, dilation=2)
#
#     pool_result1 = w5.get_pooling(img=conv_input1, pool_size=H1_after_dilation1.shape[0], stride=stride, padding=padding)
#     pool_result2 = w5.get_pooling(img=conv_input2, pool_size=H2_after_dilation2.shape[0], stride=stride, padding=padding)
#
#     print(w5.mask_convolution(img=conv_input1, mask=H1_after_dilation1, pools=pool_result1, stride=stride,
#                            padding=padding) + w5.mask_convolution(img=conv_input2, mask=H2_after_dilation2,
#                                                                pools=pool_result2, stride=stride, padding=padding))
#
#
# else:
#     pool_result1 = w5.get_pooling(img=conv_input1, pool_size= H1.shape[0], stride=stride, padding=padding)
#     pool_result2 = w5.get_pooling(img=conv_input2, pool_size= H2.shape[0], stride=stride, padding=padding)
#     final_addition = w5.mask_convolution(img=conv_input1, mask = H1, pools=pool_result1, stride=stride, padding=padding)+w5.mask_convolution(img=conv_input2, mask = H2, pools=pool_result2, stride=stride, padding=padding)
#     print(f"Final addition result: \n{final_addition}")


# ====================================================================================================

# # Find average or maximum pooling
# conv_input = np.array([
#     [0.2, 1,  0,  0.4],
#     [ -1, 0,  -0.1,  -0.1],
#     [0.1, 0,  -1,  -0.5],
#     [ 0.4, -0.7,  -0.5,  1]
# ])
#
# pooling_region = 3 # pool size, e.g. 2x2 pooling region just need to write 2
# stride = 1
# padding = 0
#
# pool_result = w5.get_pooling(img=conv_input, pool_size=pooling_region, stride= stride, padding=padding)
# print(f"Average pooling: \n {w5.average_pooling(pools=pool_result)}")
# print(f"Max pooling: \n {w5.max_pooling(pools=pool_result)}")

# ====================================================================================================
# # calculate output dimension after CNN
# # [height, width, channels] for input
# # [height, width, channels, number of mask] for mask
# input_dimension = [49, 49, 3]
# mask_dimension = [1, 1, 0,  20]
# pooling = 2
# stride = 1
# padding = 1
# use_pooling = False
#
# if not use_pooling:
#       output = w5.calculate_outputDimension(input_dimension=input_dimension, mask_dimension=mask_dimension, pooling = None, stride= stride, padding=padding)
#       print(f"Output dimension is: \n [height, width, channel]  \n {output}")
#
#       print(f"If flattering, the final length of feature vector is: {np.prod(output)}")
#
# else:
#       output =w5.calculate_outputDimension(input_dimension=input_dimension, mask_dimension=mask_dimension, pooling= pooling, stride=stride, padding=padding)
#       print(f"Output dimension is: \n [height, width, channel]  \n {output}")
#
#       print(f"If flattering, the final length of feature vector is: {np.prod(output)}")

# ====================================================================================================


"""
Week 6
- Cost function of GAN: V(D,G) -->w6.gan
- Mini-batch GAN
"""
# X=np.asarray([
#     [1,2],
#     [3,4]
# ])
# X_fake=np.asarray([
#     [5,6],
#     [7,8]
# ])
# thetas=np.asarray([0.1,0.2])
# x1, x2, t1, t2 = symbols('x1 x2 t1 t2', real=True)
# # self define discriminator function
# Dx = 1/(1+exp(-(t1*x1-t2*x2-2)))
#
# # f_d = diff(Dx, t1) # just for testing- comment out

# ==================================================================================================
# w6.gan(Dx,X,X_fake,thetas)
# w6.minibatch_GAN(num_iteration=1, k=1, learning_rate=0.02, Dx=Dx,X = X,X_fake=X_fake,thetas=thetas)

#===================================================================================================
"""
Week 7
- Karhunen-Loeve Transform (PCA)
- Hebbian Learning Rule
- Oja's Learning Rule
- Fisher's Linear Discriminant
- Extreme Learning Machine
"""
#===================================================================================================

# KLTransform(Michael Version)/PCA(Wang's Version)
S = np.asarray([
    [1,2,1],
    [2,3,1],
    [3,5,1],
    [2,2,1]
])
new_samples_to_be_classified = np.asarray([
        [3,-2,5]
])

# S = np.asarray([
#     [0,1],
#     [3,5],
#     [5,4],
#     [5,6],
#     [8,7],
#     [9,7]
# ])

# w7.KL_Transform(S = S, dimension = 2, new_samples_to_be_classified = new_samples_to_be_classified)
# w7.PCA(S= S, dimension= 2, new_samples_to_be_classified= new_samples_to_be_classified)
#===================================================================================================

# Oja's Learning Rule
# mode is either "batch" or anything else
S = np.asarray([
    [0,1],
    [3,5],
    [5,4],
    [5,6],
    [8,7],
    [9,7]
])
print(w7.oja_learning(datapoints=S, initial_weight=np.asarray([-1, 0]), learning_rate=0.01, epoch=2, mode="batch"))


