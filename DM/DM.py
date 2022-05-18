import numpy as np
import DM_Week2 as w2
import DM_Week3 as w3



"""
Week 2
Calculate error
- Sum of Squared Errors
- Mean Squared Errors
-
"""
#==================================================================

# y_hat = np.array([
#     [1.5],
#     [1.75],
#     [2.5]
# ], dtype=np.float32)
# y_values = np.array([
#     [1.6],
#     [1.5],
#     [2.4]
# ],dtype=np.float32)
#
# w2.sum_of_squared_error(y_hat, y_values)

#==================================================================
# y_hat = np.array([
#     [1.5],
#     [1.75],
#     [2.5]
# ], dtype=np.float32)
# y_values = np.array([
#     [1.6],
#     [1.5],
#     [2.4]
# ],dtype=np.float32)
#
# w2.mean_squared_error(y_hat, y_values)

#==================================================================


"""
Week 3
- Gini Coefficient
- Entropy
- Chi-Square
- Confusion Matrix
- 
"""
#==================================================================
# probability_k = np.array([
#     [0.5, 0.5],
#     [0.56, 0.44],
#     [1, 0]
# ],dtype=np.float32)
#
# w3.gini_coefficient(probability_k)

#==================================================================

# probability_k = np.array([
#     [0.5, 0.5],
#     [0.56, 0.44],
#     [1, 0]
# ],dtype=np.float32)
#
#
# w3.entropy(probability_k)

#==================================================================

# # =========Contingency Table==========
# #              |  Class 1  | Class 2 |
# #_____________________________________
# # Left Child   |      2    |     3   |
# # Right Child  |      4    |     5   |
# #_____________________________________
#
# contingency_table = np.array([
#     [2, 3,
#     4, 5],
#
#     [1, 2,
#     3, 4]
#
# ], dtype= np.float32)
# w3.chi_square(contingency_table)



#==================================================================
# # error_rate = (FP+FN)/(TP+TN+FP+FN)
# # accuracy = (TP+TN)/(TP+TN+FP+FN)
# # recall = (TP) /(TP+FN)
# # Precision = (TP) / (TP+FP)
# # f1_score = (2* recall* precision) /(recall + precision)
#
y_true = [2, 3, 3, 2, 1, 3, 2, 2, 3]
y_pred = [2, 3, 2, 3, 3, 3, 2, 2, 1]
w3.confusion_matrix_multi_class_supported(y_pred, y_true, confusion_table=None)


# if given confusion table, use below code

confusion_table_2d=np.array([
    [2, 0],
    [1, 3]
])

confusion_table_3d=np.array([
    [2, 0, 0],
    [0, 0, 1],
    [1, 0 , 2]
])
# remember to change the input of confusion table, either 2d or 3d
w3.confusion_matrix_multi_class_supported(y_pred, y_true, confusion_table=confusion_table_3d)
#==================================================================


