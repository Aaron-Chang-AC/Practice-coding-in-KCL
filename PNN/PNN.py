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

# error_rate = (FP+FN)/(TP+TN+FP+FN)
# accuracy = (TP+TN)/(TP+TN+FP+FN)
# recall = (TP) /(TP+FN)
# Precision = (TP) / (TP+FP)
# f1_score = (2* recall* precision) /(recall + precision)

# w1.confusion_matrix(
#     y_pred = np.asarray([2,3,3,2,1,3,2,2,3], dtype = np.int8),
#     y_true = np.asarray([2,3,2,3,3,3,2,2,1], dtype = np.int8),
# )

# ==============================================================================
#
# w1.knn_classifier(
#     feature_vectors = np.asarray(
#         [
#             [-2, 6],
#             [-1, -4],
#             [3, -1],
#             [-3, -2],
#             [-4, -5]
#         ]
#     ),
#     given_target = np.asarray([-2, 0]),
#     classes = np.asarray([1, 1, 1, 2, 3], dtype = np.int8),
#     k = 5,
#     euclidean_dist= True
# )

# =================================================================================
euclidean_distance = w1.euclidean_distance(x = np.asarray([2.58, -3.9]), y= np.asarray([-1, -2]))
print(f"Euclidean distance is : {0.5*(euclidean_distance**2)}")

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
 
Minimum Squared Error Procedure
- Sequential Widrowhoff
"""

# w2.classify_class_by_given_linear_discriminant_function(
#     initial_at= np.asarray([
#         [4.5, 6.7, 9.4, 6.6,1.7,2.5]
#     ], dtype=np.float32),
#     xt = np.asarray([
#          [-1, 1],
#         [1, 1],
#         [-1, 0],
#         [0.5, -0.5]
#     ]),
#     true_label= np.asarray([1, 1, 2, 2]) # keep empty np.asarray([]) if no label need to be check
# )
# ========================================================================

# (with augmented notation and sample normalisation)
# a = (w0, wt)t
# misclassified -> when g(x)<=0 !!

# w2.dichotomizer_determine_class(
#     initial_at = np.asarray([-5, 2, 1]),
#     xt = np.asarray(
#         [
#             [1,1],
#             [2,2],
#             [3,3]
#         ]
#     )
# )
# ===========================================================================

# originally g(x) > 0, change the code if question give different indication
# w2.batch_perceptron_learning(
#     initial_at = np.asarray([-25, 6, 3], dtype=float),
#     xt = np.asarray(
#         [
#             [1,5],
#             [2,5],
#             [4,1],
#             [5,1]
#         ]
#     , dtype=float),
#     true_label = np.asarray([1,1,2,2]),
#     learning_rate = 1.0,
#     epochs = 10
# )

# ==========================================================================

# originally g(x) > 0, change the code if question give different indication

#
# w2.sequential_perceptron_learning_sample_normalisation(
#     initial_at = np.asarray([-25, 6, 3], dtype=float),
#     xt = np.asarray(
#         [
#             [0,1],
#             [1,0],
#             [0.5,1.5],
#             [1.0,1.0],
#             []
#         ]
#     , dtype=float),
#     true_label = np.asarray([1,1,2,2]),
#     learning_rate = 1.0,
#     epochs = 10
# )
# ============================================================================
# basically the same as sequential perceptron learning sample normalization

# w2.sequential_perceptron_learning_using_wk(
#     initial_at = np.asarray([1,0,0], dtype=float),
#     xt = np.asarray(
#         [
#             [0,2],
#             [1,2],
#             [2,1],
#             [-3,1],
#             [-2,-1],
#             [-3,-2]
#         ]
#     , dtype=float),
#     true_label = np.asarray([1,1,1,2,2,2]),
#     learning_rate = 1.0,
#     epochs = 10
# )
# =============================================================================
# w2.sequential_multiclass_learning(
#     initial_at = np.asarray(
#         [
#             [-0.5, 0.0, 1.5],
#             [-3.0, -0.5, 0],
#             [0.5, -0.5, 0.5]
#         ]
#     , dtype=float),
#     xt = np.asarray(
#         [
#             [0, 1],
#             [1, 0],
#             [0.5, 1.5],
#             [1, 1],
#             [-0.5, 0]
#         ]
#     , dtype=float),
#     true_label = np.asarray([1,1,2,2,3]),
#     learning_rate = 1.0,
#     epochs = 1,
#     select_highest_index=False
# )
# ===================================================================================
# Update by a<-a+lr*(b_k - at_yk)*y_k
# b is the margin vector

# w2.sequential_widrow_hoff(
#     initial_at = np.asarray([1,0,0], dtype=float),
#     xt=np.asarray(
#         [
#             [0, 2],
#             [1, 2],
#             [2, 1],
#             [-3, 1],
#             [-2, -1],
#             [-3, -2]
#         ]
#     , dtype=float),
#     true_label = np.asarray([1,1,1,2,2,2]),
#     margin_vector = np.asarray([1,1,1,1,1,1], dtype=float).transpose(),
#     learning_rate = 0.1,
#     epochs = 2
# )


"""
Week 3 
- Sequential Delta Learning Rule (Supervised)
- Batch Delta Learning Rule
- Negative Feedback 
"""
# ======================================================
# H_0 = 0
# w = np.array([0.1, -0.5, 0.4])
# x1 = np.array([0.1, -0.5, 0.4])
# x2 = np.array([0.1, 0.5, 0.4])
#
# wx = w @ x2
# print(wx)
# print(w3.heaviside_function(H_0, wx))

# ======================================================
# linear threshold neuron
# w3.simple_neuron(
#     initial_theta = -2,
#     initial_w = np.asarray([0.5, 1], dtype=np.float32),
#     xt = np.asarray(
#         [
#             [0, 2],
#             [2, 1],
#             [-3, 1],
#             [-2, -1],
#             [0, -1]
#         ]
#     ),
#     H_0 = 0.5
#
# )
# ============================================================
# w3.sequential_delta_learning_rule(
#     initial_theta=-0.5,
#     initial_w = np.asarray([1, 1], dtype=np.float32),
#     xt=np.asarray(
#         [
#             [0, 0],
#             [0, 1],
#             [1, 0],
#             [1, 1],
#         ],
#         dtype=np.float32
#     ),
#     true_values = np.asarray([0, 0, 0, 1], dtype=np.float32), # true value can only be 0 or 1
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
#     epochs=7  # decision on epochs depends on the answer convergence
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
#     epochs=5
#
# )

# ================================================================

# w3.negative_feedback_network_stable(
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

"""
Week 4
- Find input weight when given all other information
- RBF
- Forward Neural Network (can be use when backpropagation is not needed)
"""
# ====================================================================================================

# input = np.array([
#     [2, -4],  # only x1
#     [-0.5, -6],  # only x2
# ])
#
# output = np.array([
#     [98, -168],  # only z1
#     [7.5, -246],  # only z2
# ])
#
# given_weight= np.array([
#     [8, -4], # weights connected to z1
#     [6, 9],   # weights connected to z2
# ])
#
# print(w4.find_weight_in_ffn(input, output, given_weight))

# ====================================================================================================

# X = np.asarray([
#     [0.1, 0.4],
#     [0.5, 0.3],
#     [0.0, 0.6],
#     [0.6, 0.3],
#     [1.0, 0.9],
#     [0.5, 0.5],
#     [0.5, 0.1],
#     [0.6, 0.8],
#     [0.9, 0.1],
#     [0.2, 0.5]
# ],dtype=np.float32)
# true_label = np.asarray([
#     [1],
#     [0],
#     [1],
#     [0],
#     [0],
#     [0],
#     [0],
#     [1],
#     [1],
#     [1]
# ],dtype=np.float32)
#
# centers = np.asarray([
#     [1, 0.9],
#     [0.2, 0.5],
#     [0.5, 0.5]
# ],dtype=np.float32)
#
# samples_to_be_classified = np.asarray([
#     [0.1],
#     [0.35],
#     [0.55],
#     [0.75],
#     [0.9]
# ],dtype=np.float32)
# # Note that the length of rho equals the number of centers
# rho = np.asarray([0.1,0.1,0.1],dtype=np.float32)
# w4.rbf_network(X = X,
#             true_label = true_label,
#             centers = centers,
#             rho = None, # if assigned rho rho = rho and change above value
#             samples_to_be_classified = samples_to_be_classified,
#             beta = 0.5,
#             function = "gaussian",
#             rho_j_method = "max", # max or average
#             bias=True
#             )
# #



# ====================================================================================================
# X = np.asarray([
#     [-0.3, 0.5]
# ],dtype=np.float32)
#
# W = {
#     'W1': np.asarray([
#         [5, 3],
#         [1, -4],
#         [-5, -2]
#     ]),
#     'B1': np.asarray([
#         [3, 3, -4]
#     ]),
#     'W2': np.asarray([
#         [3, 1, 2],
#         [-5, 3, -5]
#     ]),
#     'B2': np.asarray([
#         [-1, -1]
#     ]),
# }
# activation_function_dict = {
#     'input': w4.activation_function_spec(name='linear'),
#     'hidden1': w4.activation_function_spec(name='sigmoid'),
#     'output': w4.activation_function_spec(name='linear'),
# }
# w4.neural_network_forward(weight_dict=W, input=X, activation_function_dict=activation_function_dict)

"""
Week 5 Deep Discriminative Neural Network
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
#     [-0.5, 0.2,  0.9],
#     [ 0.5, 0.6,  0.5],
#     [-0.2, 0.4,  -0.7],
# ], dtype=np.float32)
#
# input2 = np.array([
#     [0, 0.3,  0.2],
#     [-0.4, 0.4,  -0.3],
#     [0.8, -0.8,  -0.2],
# ], dtype=np.float32)
#
# input3 = np.array([
#     [-0.6, 0.8,  -1.0],
#     [ 0.7, 0.2,  -0.9],
#     [0.9, -0.2,  0.0],
# ], dtype=np.float32)
#
#
#
# final_input_array = np.stack([input1, input2, input3])
# # print(final_input_array)
# print(w5.batch_normalization(final_input_array, beta=0.1, gamma=0.4, eta=0.2))

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
# padding= 0
# stride= 1
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
# input_dimension = [200, 300, 50]
# mask_dimension = [6, 6, 1,  40]
# pooling = 1
# stride = 2
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
Week 6 GAN
- Cost function of GAN: V(D,G) -->w6.gan
- Mini-batch GAN
"""
# X=np.asarray([
#     [1,2],
#     [3,4]
# ], dtype=np.float32)
# X_fake=np.asarray([
#     [5,6],
#     [7,8]
# ], dtype=np.float32)
# thetas=np.asarray([0.1,0.2])
# x1, x2, t1, t2 = symbols('x1 x2 t1 t2', real=True)
# # self define discriminator function
# Dx = 1/(1+exp(-(t1*x1-t2*x2-2)))

# f_d = diff(Dx, t1) # just for testing- comment out

# ==================================================================================================
# w6.gan(Dx,X,X_fake,thetas)
# w6.minibatch_GAN(num_iteration=1, k=1, learning_rate=0.02, Dx=Dx,X = X,X_fake=X_fake,thetas=thetas)

#===================================================================================================
"""
Week 7 Feature Extraction
- Karhunen-Loeve Transform (PCA)
- Hebbian Learning Rule
- Oja's Learning Rule
# Linear Discriminant Analysis
- Fisher's Linear Discriminant
- Extreme Learning Machine
- Sparsity
"""
#===================================================================================================

# KLTransform(Michael Version)/PCA(Wang's Version)
# S = np.asarray([
#     [1, 2],
#     [3,5],
#     [5,4],
#     [8,7],
#     [11,7]
# ])
# new_samples_to_be_classified = np.asarray([
#     [1, 2],
#     [3, 5],
#     [5, 4],
#     [8, 7],
#     [11, 7]
# ])

# S = np.asarray([
#     [5.0, 5.0, 4.4, 3.2],
#     [6.2, 7.0, 6.3, 5.7],
#     [5.5, 5.0, 5.2, 3.2],
#     [3.1, 6.3, 4.0, 2.5],
#     [6.2, 5.6, 2.3, 6.1],
# ])


# w7.KL_Transform(S = S, dimension = 1, new_samples_to_be_classified = new_samples_to_be_classified)
# w7.PCA(S= S, dimension= 2, new_samples_to_be_classified= new_samples_to_be_classified)

#=====================================================================================
## if given eigenvalues and eigenvectors
# eigenvalues = np.array([0.00, 0.71, 1.90, 3.21])
# eigenvectors = np.array([
#     [-0.59, 0.55, 0.11, 0.58],
#     [-0.56, -0.78, 0.25, 0.12],
#     [0.25, 0.12, 0.96, -0.04],
#     [0.52, -0.27, -0.07, 0.81]
# ])
# new_samples_to_be_classified = np.asarray([
#         [5.0, 5.0, 4.4, 3.2]
# ])
# dimension = 2
# X = S.copy().T
# m = X.mean(axis=1)
# # don't change below code if no extra circumstances
# idx = eigenvalues.argsort()[::-1]
# W = eigenvalues[idx]
# V = eigenvectors[:, idx]
#
# print(f"Eigenvalues (Sorted):\n{W}\n")
# print(f"Eigenvectors(Sorted):\n{V}\n")
#
# V_hat = V.copy()[:, 0:dimension]
# V_hatT = V_hat.T
# print(f"V_hatT:\n{V_hatT}\n")
#
# if len(new_samples_to_be_classified) > 0:
#     new_m = new_samples_to_be_classified - m
#     print(new_m)
#     new_targets = V_hatT @ (new_m.T)
#     print(f"results(each column is a converted sample):\n{new_targets}\n")
#


#===================================================================================================

# Oja's Learning Rule
# mode is either "batch" or anything else
# will give zero-mean dataset as well

# Note that given_zero_mean is a flag which should be true when given
# dataset is zero mean

# S = np.asarray([
#     [0,1],
#     [3,5],
#     [5,4],
#     [5,6],
#     [8,7],
#     [9,7]
# ],dtype=np.float32)
# w7.oja_learning(datapoints=S, given_zero_mean=False, initial_weight=np.asarray([-1, 0]), learning_rate=0.01, epoch=2, mode="batch")

#===================================================================================================
# Hebbian Learning Rule
# for hebbian_learning
# Note that given_zero_mean is a flag which should be true when given
# dataset is zero mean
# S = np.asarray([
#     [0,1],
#     [3,5],
#     [5,4],
#     [5,6],
#     [8,7],
#     [9,7]
# ],dtype=np.float32)
# initial_weight=np.asarray([-1, 0],dtype=np.float32)
# w7.hebbian_learning(datapoints=S, given_zero_mean=True, initial_weight=initial_weight, learning_rate=0.01, epoch=6, mode="batch")

#===================================================================================================
# For LDA
# support multi-dimension
# datapoints = np.array([
#     [6, 0],
#     [0, 4],
#     [5, -2],
#     [4, -4]
# ], dtype=np.float32)
#
# classes = np.array([1, 1, 2, 2])
# projection_vector = np.array([
#     [1, 1]
# ], dtype=np.float32)
# w7.LDA(datapoints=datapoints, classes= classes, projection_vector=projection_vector)
#===================================================================================================

# # LDA
# # Fisher's Method
# datapoints = np.asarray([[1, 2], [2, 1],[3, 3], [6, 5], [7, 8]])
# classes = np.asarray([1, 1, 1, 2, 2])
# projection_weight = np.asarray([[-1, 5], [2, -6]])
# print(w7.fisher_method(datapoints, classes, projection_weight=projection_weight))

#===================================================================================================

# Extreme Learning Machine
# weight = np.asarray([0,0,0,-1,0,0,2]) # output neuron weight
# V = np.asarray([[-0.62, 0.44, -0.91],
#                 [-0.81, -0.09, 0.02],
#                 [0.74, -0.91, -0.60],
#                 [-0.82, -0.92, 0.71],
#                 [-0.26, 0.68, 0.15],
#                 [0.8, -0.94, -0.83]])
# datapoints = np.asarray([[0,0], [0, 1],[1, 0], [1, 1]])
#
# w7.extreme_learning_machine(V, weight, datapoints, H_0=0.5)

#===================================================================================================

# # Sparse Coding
# # ||x-(V^t)y||+lambda||y||
# V = np.asarray([[0.4, -0.6],
#               [0.55, -0.45],
#               [0.5, -0.5],
#               [-0.1, 0.9],
#               [-0.5, -0.5],
#               [0.9, 0.1],
#               [0.5, 0.5],
#               [0.45, 0.55]])
# x = np.asarray([-0.05, -0.95])
# y1_t = np.asarray([1, 0, 0, 0, 1, 0, 0, 0])
# y2_t = np.asarray([0, 0, 1, 0, 0, 0, -1, 0])
# y3_t = np.asarray([0, 0, 0, -1, 0, 0, 0, 0])
# y_t = np.asarray([y1_t, y2_t, y3_t])
# w7.sparse_coding(V, x, y_t, LAMBDA=0.0)

#===================================================================================================
"""
Week 8 Support Vector Machines
- Hyperplane
- Margin of hyperplane
- Plot and find support vector
"""
# svs = [
#     [3, 3],
#     [-1, -2]
# ]
# # class only 1 or -1
# classes = [1,-1]
# hyperplane = w8.LinearSVM(svs, classes)
#
# # Margin given by hyperplane
# w8.margin_from_hyperplane(hyperplane)


#===================================================================================================

# svs = [
#     [1,1],
#     [1,-1],
#     [-1,1],
#     [-1,-1]
# ]
# classes = [1,1,-1,-1]
# w8.show_points(svs,classes)
"""
Week 9 Support Vector Machines

- Adaboost
- Bagging Algorithm 
"""
#===================================================================================================
# for adaboost
# X = np.asarray([[1,0],[-1,0],[0,1],[0,-1]])
# y = np.asarray([1,1,-1,-1])
# table = np.asarray([
#     [1,-1,1,1],
#     [-1,1,-1,-1],
#     [1,-1,-1,-1],
#     [-1,1,1,1],
#     [1,1,1,-1],
#     [-1,-1,-1,1],
#     [-1,-1,1,-1],
#     [1,1,-1,1]
# ])
# print(w9.adaboost(len(table), X, y, table, select_highest = False))

#===================================================================================================

# for bagging
# y = np.asarray([1,1,-1,-1])
# table = np.asarray([
#     [1,-1,1,1],
#     [-1,1,-1,-1],
#     [1,-1,-1,-1],
#     [-1,1,1,1],
#     [1,1,1,-1],
#     [-1,-1,-1,1],
#     [-1,-1,1,-1],
#     [1,1,-1,1]
# ])
# w9.bagging_algo(y, table)

#===================================================================================================

"""
Week 10 Clustering

- K-Means
- Competitive Learning Algorithm
- Fuzzy K-Means
- Agglomerative Clustering
-  
"""
from sklearn.metrics.pairwise import euclidean_distances

#===================================================================================================

# for k_means
# Note that each "row" is a sample, c is the number of clusters
# cluster_point is the initial clusters
# datapoint = np.asarray([
#     [4, 5],
#     [4, 10],
#     [1, 5],
#     [2, 1],
#     [0, 9],
#     [0, 6]
# ])
# cluster_point=np.asarray([
#     [1, 5],
#     [4, 10]
# ])
# # mode can be either "euclidean" or in "manhattan"
# w10.k_means(datapoint=datapoint, c=2, cluster_point=cluster_point, randomized=False, mode="euclidean")
#

#===================================================================================================


# competitive_learning_algorithm
# Note that each integer in chosen_order is >= 0
# S = np.asarray([
#     [-1, 3],
#     [1, 4],
#     [0, 5],
#     [4, -1],
#     [3, 0],
#     [5, 1]
# ], dtype=np.float32)
# initial_centers=np.asarray([
#     [-0.5, 1.5],
#     [0, 2.5],
#     [1.5, 0]
# ], dtype=float)
# # remember chosen order need to start from 0 because of the design of code
# chosen_order=np.asarray([2, 0, 0, 4, 5]) # if no chosen order then 0,1,2,3,....
# new_data_to_be_classfied = np.asarray([
#     [0,-2]
# ])
# w10.competitive_learning_algorithm(S,iterations=5,initial_centers= initial_centers,
#                                chosen_order=chosen_order,
#                                new_data_to_be_classfied=new_data_to_be_classfied,
#                                learning_rate=0.1,
#                                normalization_flag=False
#                                )


#===================================================================================================

# basic_leader_follower_algorithm
# S = np.asarray([
#     [-1,3],
#     [1,4],
#     [0,5],
#     [4,-1],
#     [3,0],
#     [5,1]
# ], dtype=np.float32)
# chosen_order=np.asarray([2,0,0,4,5])
# initial_centers=np.asarray([
#     S[chosen_order[0]].tolist()
# ], dtype=np.float32)
# new_data_to_be_classfied = np.asarray([
#     [0,-2],
#     [-0.2,2.8],
#     [0,1.5]
# ])
# w10.basic_leader_follower_algorithm(S,iterations=5,initial_centers=initial_centers,
#                                 chosen_order=chosen_order,
#                                 new_data_to_be_classfied=new_data_to_be_classfied,
#                                 learning_rate=0.5,
#                                 theta=3.0,
#                                 normalization_flag=False
#                                 )

#===================================================================================================

# for fuzzyKMeans
# each row is a sample
# dataset = np.asarray([
#     [-1, 3],
#     [1, 4],
#     [0, 5],
#     [4, -1],
#     [3, 0],
#     [5, 1]
# ])
# u = np.asarray([
#     [1, 0],
#     [0.5, 0.5],
#     [0.5, 0.5],
#     [0.5, 0.5],
#     [0.5, 0.5],
#     [0, 1]
# ])
# w10.fuzzyKMeans(dataset=dataset, numCluster=2, initial_membership=u, b=2, criteria=0.5)

#===================================================================================================

# for Agglomerative_clustering
# hdataset = [
#     [1, 0],
#     [0, 2],
#     [1, 3],
#     [3, 0],
#     [3, 1]
# ]

## ord 1 --> manhattan /////  ord 2 --> euclidean
# w10.Agglomerative_clustering(dataset= hdataset, numCluster= 2, link_type="single", ord=2)
# w10.Agglomerative_clustering(dataset= hdataset, numCluster= 3, link_type="complete", ord=2)
# w10.Agglomerative_clustering(dataset= hdataset, numCluster= 3, link_type="average", ord=2)
# w10.Agglomerative_clustering(dataset= hdataset, numCluster= 3, link_type="mean", ord=2) # so-called centroid
##########################Checking using sklearn#########################
# from sklearn.cluster import AgglomerativeClustering
#
# X = np.asarray([
#     [-1, 3],
#     [1, 2],
#     [0, 1],
#     [4, 0],
#     [5, 4],
#     [3, 2]
# ])
# clustering = AgglomerativeClustering(n_clusters=3, linkage="average").fit(X)
# print(clustering.labels_)

# print(euclidean_distance(np.asarray([[-2.8284, 0],[2.8284, 0]]), np.asarray([-0.7071, -3.5355])))

#===================================================================================================

# for iterative optimization
# datapoint = np.array([
#     [2, 3],
#     [3, 2],
#     [4, 3],
#     [6, 3],
#     [8, 2],
#     [9, 3],
#     [10, 1]
# ])
# # initial_datapoint_class = np.asarray([0, 0, 0, 1, 1, 1])
# cluster_point=np.asarray([
#     [4.000, 1.000],
#     [7.000, 1.000]
# ])
# initial_clustering = np.array([0, 0, 0, 1, 1, 1, 1]) # use k-mean to obtain this
# selected_datapoint_tomove = np.array([[6,3]])
# w10.iterative_optimization(datapoint=datapoint, initial_clustering= initial_clustering, clusters=cluster_point, selected_datapoint_tomove= selected_datapoint_tomove)
