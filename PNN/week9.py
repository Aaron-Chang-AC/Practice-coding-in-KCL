import numpy as np


def adaboost(k_max, dataset, output_class, classfier_table):
    n = len(dataset) #4
    n_classfier_table = len(classfier_table)
    W = np.ones(n) # [1 , 1, 1 ,1 ]
    W = W/n #[0.25 , 0.25, 0.25, 0.25]
    training_error=np.zeros(n_classfier_table)
    alpha = np.zeros(n_classfier_table)
    hk_list = np.zeros(n_classfier_table)

    k = 0
    for i in range(0, k_max):
        print(f"======================Round {i+1}========================")
        for j in range(n_classfier_table):
            training_error[j]=0
            for l in range(n):
                if classfier_table[j,l] != output_class[l]:
                    training_error[j] += W[l]
        print(training_error)

        hk_list[i] = np.argmin(training_error)
        ek = np.min(training_error)
        print(f"Overall weighted error rate:{ek}")
        if ek > 0.5:
            k_max = k-1
            print("Process done with ek > 0.5")
            break

        alpha[i] = 0.5 * np.log((1 - ek) / ek)

        for j in range(n):
            W[j] = W[j]*np.exp(-alpha[i]*output_class[j]*classfier_table[int(hk_list[i]),j])
        Z = np.sum(W)

        W = W / Z
        print(f"Zk is {Z}")
        print(f"W k+1 is {W}")
        print(f"Alpha is {alpha[i]}")
        k+=1

        temp = np.zeros(n)
        for j in range(n):
            print("the ", j, "th sample:")
            for m in range(i + 1):
                temp[j]+=alpha[m]*classfier_table[int(hk_list[m]),j]
            # print(temp[j])
        temp[temp>=0]=1
        temp[temp<0]=-1
        # print(temp)

        if np.array_equal(output_class, temp):
            print(f"Adaboost Classifier found")
            break # if condition different comment out


X = np.asarray([[1,0],[-1,0],[0,1],[0,-1]])
y = np.asarray([1,1,-1,-1])
table = np.asarray([
    [1,-1,1,1],
    [-1,1,-1,-1],
    [1,-1,-1,-1],
    [-1,1,1,1],
    [1,1,1,-1],
    [-1,-1,-1,1],
    [-1,-1,1,-1],
    [1,1,-1,1]
])
print(adaboost(len(table), X, y, table))


def bagging_algo(output_class, classfier_table):
    result = classfier_table.copy().sum(axis=0)
    result[result>=0]=1
    result[result<0]=-1
    training_error=0.0
    n = len(output_class)
    for i in range(n):
        if result[i] != output_class[i]:
            training_error += (1.0/n)
    print(f"result:{result}, training error is {training_error}")

bagging_algo(y, table)