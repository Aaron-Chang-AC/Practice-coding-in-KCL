import numpy as np


def gini_coefficient(probability_k):
    result_list = []
    for i in range(len(probability_k)):
        squared_prob_list = []
        for j in range(len(probability_k[i])):
            squared_prob = (probability_k[i][j])**2
            squared_prob_list.append(squared_prob)
        print(f"G(L_{i+1}): {sum(squared_prob_list)}")
        result_list.append(sum(squared_prob_list))
    print("The higher result of the coefficient, more purity")
    print(f"The final decision is to pick G(L_{np.argmax(result_list)+1})")

    return


np.seterr(all='raise') # debugging the Runtime warning raised from log(0)
def entropy(probability_k):
    print("The lower result of the entropy, more purity")
    result_list = []

    for i in range(len(probability_k)):
        entropy_prob_list = []
        for j in range(len(probability_k[i])):
            try:
                entropy_prob = -1*((probability_k[i][j])*np.log2(probability_k[i][j]))
            except:
                entropy_prob = 0
            entropy_prob_list.append(entropy_prob)
        print(f"E(L_{i+1}): {sum(entropy_prob_list)}")
        result_list.append(sum(entropy_prob_list))
    print("The lower result of the entropy, more purity")
    print(f"The final decision is to pick E(L_{np.argmin(result_list)+1})")

    return


def chi_square(contingency_table):
    print("The higher result of the chi-square, children have more purity")
    result_list = []

    for c in range(len(contingency_table)):
        print("=================Next Class===========================")
        chi_square_prob_list = []
        O_k = contingency_table[c][0]
        print(f"Ok is {O_k}")
        E_k = (O_k+contingency_table[c][1])*((O_k + contingency_table[c][2] )/sum(contingency_table[c]))
        print(f"Ek is {E_k}")
        chi_square_prob = (O_k - E_k) ** 2 / E_k
        chi_square_prob_list.append(chi_square_prob)
        print(f"C(L_{c+1}): {sum(chi_square_prob_list)}\n")
        result_list.append(sum(chi_square_prob_list))
    print("The higher result of the chi-square, children have more purity")
    print(f"The final decision is to pick C(L_{np.argmax(result_list)+1})")

    return


from sklearn.metrics import confusion_matrix
def confusion_matrix_multi_class_supported(y_pred,y_true, labels=None, confusion_table=None):
    if confusion_table is None:
        if labels is None:
            cm = confusion_matrix(y_true, y_pred)
        else:
            cm = confusion_matrix(y_true, y_pred, labels=labels)
        print(f"Confusion Matrix is: \n {cm}")
    else:
        cm = confusion_table
        print(f"Confusion Matrix is: \n {cm}")

    try:
        tn, fp, fn, tp = cm.ravel()
        print(f"True Positive (TP): {tp}")
        print(f"True Negative (TN): {tn}")
        print(f"False Positive (FP): {fp}")
        print(f"False Negative (FN): {fn}")

        print("error-rate:", (fp + fn) / (tp + fp + tn + fn))
        print("accuracy:", (tp + tn) / (tp + fp + tn + fn))
        print("recall:", tp / (tp + fn))
        print("precision:", tp / (tp + fp))
        print("f1-score", (2 * tp) / (2 * tp + fp + fn))
    except ValueError:
        tp = cm[0][0] + cm[1][1] + cm[2][2]
        fp = cm[0][1] + cm[0][2]+ cm[1][0] + cm[1][2]+cm[2][0] + cm[2][1]
        tn = 0
        fn = 0
        print(f"True Positive (TP): {tp}")
        print(f"True Negative (TN): {tn}")
        print(f"False Positive (FP): {fp}")
        print(f"False Negative (FN): {fn}")

        print("error-rate:", (fp + fn) / (tp + fp + tn + fn))
        print("accuracy:", (tp + tn) / (tp + fp + tn + fn))
    print("\n")


# y_true = [2, 0, 2, 2, 0, 1]
# y_pred = [0, 0, 2, 2, 0, 2]
# confusion_table=np.array([
#     [2, 0, 0],
#     [0, 0, 1],
#     [1, 0 , 2]
# ])
# confusion_matrix_multi_class_supported(y_pred, y_true, confusion_table=confusion_table)