import numpy as np
import random
from scipy.spatial.distance import cityblock

import pandas as pd


def k_means(datapoint, c, cluster_point=None, randomized=False, mode="euclidean"):
    data = datapoint.copy()
    num_cluster = c
    clusters = []
    if mode == "euclidean":
        print("Euclidean metrics is in use")
    else:
        print("Manhattan metrics is in use")
    if randomized == True:
        for i in range(num_cluster):
            clusters.append(random.choice(data))
            data.remove(clusters[i])
    else:
        # define own m1 and m2 in the list if not randomized
        clusters = np.asarray(cluster_point)

    assigned_class_list = []
    assigned_new_class_list = []
    new_coordinates = np.zeros((num_cluster,len(data[0])))
    new_coordinates_cnt = np.zeros(num_cluster)

    euclidean = []
    round = 1
    while True:
        print(f"======================Round {round} =================================")
        for i in range(len(datapoint)):
            for j in range(len(clusters)):
                if mode=="euclidean":
                    result = np.linalg.norm(datapoint[i] - clusters[j])
                    euclidean.append(result)
                else:
                    result = cityblock(datapoint[i], clusters[j])
                    euclidean.append(result)
                # print(result)
            print("-------Next sample-------")
            print(f"euclidean distances: {euclidean}")
            print(f"Argmin is {np.argmin(euclidean)}")
            assigned_new_class_list.append(np.argmin(euclidean))
            print(f"Assigned Class List: {assigned_new_class_list}")

            euclidean[:] = [] # empty list

        if round==1:
            assigned_class_list = assigned_new_class_list.copy()

            for j in range(len(assigned_class_list)):
                new_coordinates[assigned_class_list[j]] += datapoint[j]
                new_coordinates_cnt[assigned_class_list[j]] += 1

            for j in range(num_cluster):
                if int(new_coordinates_cnt[j]) == 0:
                    new_coordinates[j] = clusters[j]
                else:
                    new_coordinates[j] /= int(new_coordinates_cnt[j])

            clusters = new_coordinates.copy()
            print(f"New cluster is: \n {clusters}")
            assigned_new_class_list[:] = []
        else:
            if assigned_class_list == assigned_new_class_list:
                print(f"Proccess Done in Round {round}\n\n")
                break
            else:
                assigned_class_list = assigned_new_class_list.copy()

                for j in range(len(assigned_class_list)):
                    new_coordinates[assigned_class_list[j]] += datapoint[j]
                    new_coordinates_cnt[assigned_class_list[j]] += 1
                for j in range(num_cluster):
                    if int(new_coordinates_cnt[j]) == 0:
                        new_coordinates[j] = clusters[j]
                    else:
                        new_coordinates[j] /= int(new_coordinates_cnt[j])
                clusters = new_coordinates.copy()
                assigned_new_class_list[:] = []
        new_coordinates = np.zeros((num_cluster, len(data[0])))
        new_coordinates_cnt = np.zeros(num_cluster)
        round += 1
    for i in range(len(assigned_class_list)):
        print(f"X{i + 1} belongs to class {assigned_class_list[i]+1}")
    for i in range(len(clusters)):
        print(f"cluster {i + 1}: {clusters[i]}")
    return clusters

def PCA(S, dimension, new_samples_to_be_classified):
    n = len(S)
    X = S.copy().T
    print(f"original dataset(each column is a sample):\n{X}\n")
    m = X.mean(axis=1)
    print(f"mean:\n{m.reshape(-1,1)}\n")
    Xi_m= X-m.reshape(-1,1)
    covariance_matrix=(Xi_m @ Xi_m.T)/n
    print(f"covariance_matrix:\n{covariance_matrix}\n")
    V, D, VT = np.linalg.svd(covariance_matrix, full_matrices=True)
    idx = (-D).argsort()
    D = np.diag(D[idx])
    V = V[:,idx]
    VT = V.T
    print(f"V:\n{V}\n")
    print(f"D:\n{D}\n")
    print(f"VT:\n{VT}\n")

    V_hat = V.copy()[:,0:dimension]
    V_hatT = V_hat.T
    print(f"V_hatT:\n{V_hatT}\n")
    results = V_hatT @ Xi_m
    print(f"results(each column is a converted sample):\n{results}\n")

    if len(new_samples_to_be_classified)>0:
        new_targets=V_hatT @ new_samples_to_be_classified.T
        print(f"new_targets(each column is a converted sample):\n{new_targets}\n")

def competitive_learning_algorithm(S,iterations,initial_centers,chosen_order,new_data_to_be_classfied,learning_rate,normalization_flag):
    '''
     chosen_order: A np array which should start from 0, and it contains the indices that we
     can use to select the samples from dataset X

     new_data_to_be_classfied: A np array which contains the new given datapoints, if not given
     in the question, just manually give a legal datapoint

    '''
    n = len(S)
    n_centers = len(initial_centers)
    n_new_samples = len(new_data_to_be_classfied)
    X = np.ones((n, 1), dtype=np.float32)
    m = np.ones((n_centers, 1), dtype=np.float32)
    target = np.ones((n_new_samples, 1), dtype=np.float32)

    if normalization_flag:
        X = np.append(X, S.copy(), axis=1)
        m = np.append(m, initial_centers.copy(), axis=1)
        target = np.append(target, new_data_to_be_classfied.copy(), axis=1)
        for i in range(n):
            X[i] = X[i] / np.linalg.norm(X[i])
        for i in range(n_centers):
            m[i] = m[i] / np.linalg.norm(m[i])
        for i in range(n_new_samples):
            target[i] = target[i] / np.linalg.norm(target[i])
        X = X.T
        m = m.T
        target = target.T
    else:
        X = S.copy().T
        m = initial_centers.copy().T
        target = new_data_to_be_classfied.copy().T

    print(f"original dataset(each column is a sample):\n{X}\n")
    print(f"initial centers(each column is a center):\n{m}\n")

    if normalization_flag:
        for i in range(iterations):
            print(f"-------------Iteration {i + 1}-------------\n")
            sample_x = X[:, chosen_order[i]]
            dist = np.zeros(n_centers)

            for j in range(n_centers):
                dist[j] = np.dot(sample_x.flatten(), m[:, j].flatten())
                print(f"dot product of weight {j + 1}: {dist[j]}\n")

            update_index = np.argmax(dist)
            m[:, update_index] = m[:, update_index] + learning_rate * sample_x
            print(f"updated weight {update_index + 1}:\n{m[:, update_index]}\n")
            m[:, update_index] = m[:, update_index] / np.linalg.norm(m[:, update_index])
            print(f"normalize weight {update_index + 1}:\n{m[:, update_index]}\n")

        print(f"all updated weights:(each column is a weight)\n{m}\n")

        print(f"assigned classes to all original samples:")
        for i in range(n):
            dist = np.zeros(n_centers)
            sample_x = X[:, i]
            for j in range(n_centers):
                dist[j] = np.dot(sample_x.flatten(), m[:, j].flatten())
            print(f"sample {i + 1}: class {np.argmax(dist) + 1}")

        print(f"\n\nassigned classes to all new given samples:")
        for i in range(n_new_samples):
            dist = np.zeros(n_centers)
            sample_x = target[:, i]
            for j in range(n_centers):
                dist[j] = np.dot(sample_x.flatten(), m[:, j].flatten())
            print(f"sample {i + 1}: class {np.argmax(dist) + 1}")

    else:
        for i in range(iterations):
            print(f"-------------Iteration {i + 1}-------------\n")
            sample_x = X[:, chosen_order[i]]
            dist = np.zeros(n_centers)

            for j in range(n_centers):
                dist[j] = np.linalg.norm(sample_x - m[:, j])
                print(f"distance to center {j + 1}: {dist[j]}\n")

            update_index = np.argmin(dist)
            m[:, update_index] = m[:, update_index] + learning_rate * (sample_x - m[:, update_index])
            print(f"updated center {update_index + 1}:\n{m[:, update_index]}\n")

        print(f"all updated centers:(each column is a center)\n{m}\n")

        print(f"assigned classes to all original samples:")
        for i in range(n):
            dist = np.zeros(n_centers)
            sample_x = X[:, i]
            for j in range(n_centers):
                dist[j] = np.linalg.norm(sample_x - m[:, j])
            print(f"sample {i + 1}: class {np.argmin(dist) + 1}")

        print(f"\n\nassigned classes to all new given samples:")
        for i in range(n_new_samples):
            dist = np.zeros(n_centers)
            sample_x = target[:, i]
            for j in range(n_centers):
                dist[j] = np.linalg.norm(sample_x - m[:, j])
            print(f"sample {i + 1}: class {np.argmin(dist) + 1}")

def basic_leader_follower_algorithm(S,iterations,initial_centers,chosen_order,new_data_to_be_classfied,learning_rate,theta,normalization_flag):
    '''
     chosen_order: A np array which should start from 0, and it contains the indices that we
     can use to select the samples from dataset X

     new_data_to_be_classfied: A np array which contains the new given datapoints, if not given
     in the question, just manually give a legal datapoint

    '''
    n = len(S)
    n_centers = len(initial_centers)
    n_new_samples = len(new_data_to_be_classfied)
    X = np.ones((n, 1), dtype=np.float32)
    m = np.ones((n_centers, 1), dtype=np.float32)
    target = np.ones((n_new_samples, 1), dtype=np.float32)

    if normalization_flag:
        X = np.append(X, S.copy(), axis=1)
        m = np.append(m, initial_centers.copy(), axis=1)
        target = np.append(target, new_data_to_be_classfied.copy(), axis=1)
        for i in range(n):
            X[i]=X[i]/np.linalg.norm(X[i])
        for i in range(n_centers):
            m[i] = m[i] / np.linalg.norm(m[i])
        for i in range(n_new_samples):
            target[i] = target[i] / np.linalg.norm(target[i])
        X = X.T
        m = m.T
        target = target.T
    else:
        X = S.copy().T
        m = initial_centers.copy().T
        target = new_data_to_be_classfied.copy().T

    print(f"original dataset(each column is a sample):\n{X}\n")
    print(f"initial centers(each column is a center):\n{m}\n")
    if normalization_flag:
        for i in range(iterations):
            print(f"-------------Iteration {i + 1}-------------\n")
            sample_x = X[:, chosen_order[i]]
            dist = np.zeros(n_centers)

            for j in range(n_centers):
                dist[j] = np.dot(sample_x.flatten(), m[:, j].flatten())
                print(f"dot product of weight {j + 1}: {dist[j]}\n")

            best_weight_idx = np.argmax(dist)
            temp = np.linalg.norm(sample_x - m[:, best_weight_idx])
            print(f"distance of the best cluster to the sample: {temp}")
            if temp < theta:
                m[:, best_weight_idx] = m[:, best_weight_idx] + learning_rate * sample_x
                print(f"updated center {best_weight_idx + 1}:\n{m[:, best_weight_idx]}\n")
                m[:, best_weight_idx] = m[:, best_weight_idx] / np.linalg.norm(m[:, best_weight_idx])
                print(f"normalize center {best_weight_idx + 1}:\n{m[:, best_weight_idx]}\n")
            else:
                n_centers += 1
                print(f"original closest center {best_weight_idx + 1}:\n{m[:, best_weight_idx]}\n")
                m = np.append(m, np.asarray([sample_x.copy()]).T, axis=1)
                m[:, n_centers - 1] = m[:, n_centers - 1] / np.linalg.norm(m[:, n_centers - 1])
                print(f"new center {n_centers}:\n{m[:, n_centers - 1]}\n")

        print(f"all updated centers:(each column is a center)\n{m}\n")

        print(f"assigned classes to all original samples:")
        for i in range(n):
            dist = np.zeros(n_centers)
            sample_x = X[:, i]
            for j in range(n_centers):
                dist[j] = np.dot(sample_x.flatten(), m[:, j].flatten())
            print(f"sample {i + 1}: class {np.argmax(dist) + 1} weight:{np.max(dist)}")

        print(f"\n\nassigned classes to all new given samples:")
        for i in range(n_new_samples):
            dist = np.zeros(n_centers)
            sample_x = target[:, i]
            for j in range(n_centers):
                dist[j] = np.dot(sample_x.flatten(), m[:, j].flatten())
            print(f"sample {i + 1}: class {np.argmax(dist) + 1} weight:{np.max(dist)}")
    else:
        for i in range(iterations):
            print(f"-------------Iteration {i + 1}-------------\n")
            sample_x = X[:, chosen_order[i]]
            dist = np.zeros(n_centers)

            for j in range(n_centers):
                dist[j] = np.linalg.norm(sample_x - m[:, j])
                print(f"distance to center {j + 1}: {dist[j]}\n")

            if np.min(dist) < theta:
                update_index = np.argmin(dist)
                m[:, update_index] = m[:, update_index] + learning_rate * (sample_x - m[:, update_index])
                print(f"updated center {update_index + 1}:\n{m[:, update_index]}\n")
            else:
                n_centers += 1
                min_dist_index = np.argmin(dist)
                print(f"original closest center {min_dist_index + 1}:\n{m[:, min_dist_index]}\n")
                m = np.append(m, np.asarray([sample_x.copy()]).T, axis=1)
                print(f"new center {n_centers}:\n{m[:, n_centers - 1]}\n")

        print(f"all updated centers:(each column is a center)\n{m}\n")

        print(f"assigned classes to all original samples:")
        for i in range(n):
            dist = np.zeros(n_centers)
            sample_x = X[:, i]
            for j in range(n_centers):
                dist[j] = np.linalg.norm(sample_x - m[:, j])
            print(f"sample {i + 1}: class {np.argmin(dist) + 1} distance:{np.min(dist)}")

        print(f"\n\nassigned classes to all new given samples:")
        for i in range(n_new_samples):
            dist = np.zeros(n_centers)
            sample_x = target[:, i]
            for j in range(n_centers):
                dist[j] = np.linalg.norm(sample_x - m[:, j])
            print(f"sample {i + 1}: class {np.argmin(dist) + 1} distance:{np.min(dist)}")


def fuzzyKMeans(dataset, numCluster, initial_membership, b, criteria):
    """

    :param dataset:
    :param numCluster:
    :param initial_membership: u (greek letter mu)
    :return:
    """
    # normalized_membership
    normalized_membership = []
    new_pair = []
    for i in range(len(initial_membership)):
        for j in range(len(initial_membership[0])):
            new_member = initial_membership[i][j] / np.sum(initial_membership[i])
            new_pair.append(new_member)
        normalized_membership.append(new_pair.copy())
        new_pair[:] = []

    normalized_membership = np.asarray(normalized_membership.copy())
    print(f"Normalized Membership is:\n{normalized_membership}")
    print(f"Transpose of Normalized Membership is:\n{normalized_membership.T}\n\n")

    #cluster point
    cluster_point = np.zeros((numCluster,len(dataset[0])))
    old_cluster_point = np.zeros((numCluster,len(dataset[0])))
    round=1
    euclidean = []
    while True:
        print(f"=====Round {round} started======")
        cluster_point = np.zeros((numCluster, len(dataset[0])))
        # update cluster centers
        for i in range(numCluster): # number of clusters
            for j in range(len(dataset)):
                cluster_point[i] += np.multiply(normalized_membership.T[i][j]**b, dataset[j])
            cluster_point[i] /= np.sum(normalized_membership.T[i]**b)

        for i in range(numCluster):
            print(f"updated m{i+1}: {cluster_point[i]}")
        print("\n")
        TEMP = normalized_membership.copy().T
        for i in range(numCluster): # number of clusters
            for j in range(len(dataset)):
                dist =  np.linalg.norm(dataset[j] - cluster_point[i])
                dist = dist**(-2/(b-1))
                TEMP[i,j] = dist
        TEMP = TEMP / np.sum(TEMP, axis=0)
        normalized_membership = TEMP.copy().T
        print(f"updated normalized_membership:\n{normalized_membership.T}\n")

        if round == 1:
            old_cluster_point = cluster_point.copy()
        else:
            diff_list = []
            for i in range(len(old_cluster_point)):
                for j in range(len(dataset[0])):
                    diff = np.abs(old_cluster_point[i][j] - cluster_point[i][j])
                    diff_list.append(diff)
            print(f"difference list: {diff_list}")
            if all(i<0.5 for i in diff_list):
                break

            else:
                old_cluster_point = cluster_point.copy()
        round+=1


from sklearn.metrics.pairwise import euclidean_distances
def agglomerative_clustering(dataset, numCluster, link_type=None):
    # link_type--> "single", "complete", "average" (還有complete 跟average)
    distance_matrix = euclidean_distances(dataset, dataset)
    distance_matrix = np.tril(distance_matrix)
    distance_matrix[distance_matrix==0] = np.inf
    print(distance_matrix)
    df = pd.DataFrame(data=np.ones(dataset.shape[0])*np.inf) # initialized dataframe
    # error handling if numCluster is over needed
    if numCluster > distance_matrix.shape[0]:
        print("Reconsider the number of clusters!")
        numCluster = distance_matrix.shape[0]

    if link_type == "single":
        d = {}  # This dictionary keeps record of which data points or cluster are merging
        for i in range(0, numCluster):
            print(f"=======Iteration {i+1} Started========")
            print(f"The minimum value founded is {np.min(distance_matrix)}")
            #argmin returns the indexes of the first occureneces of the minimum values in flattened matrix
            ij_min = np.unravel_index(distance_matrix.argmin(),
                                      distance_matrix.shape)  # from the distance matrix, get the minimum distance
            # np.unravel_index gives us the position of minimum distance. e.g. (1,2) and (0,1) is where minimum value is present in matrix.
            # This is what we need as in Hierarchical clustering, we merge the two pairs with minimum distance
            if i == 0:
                df.iloc[ij_min[0]] = 0
                df.iloc[ij_min[1]] = 0
                print(f"df is {df}")
            else:
                try:
                    a = int(df.iloc[ij_min[0]])

                except:
                    df.iloc[ij_min[0]] = i
                    a = i

                try:
                    b = int(df.iloc[ij_min[1]])
                except:
                    df.iloc[ij_min[1]] = i
                    b = i
                df[(df[0] == a) | (df[0] == b)] = i
                print(df)
            d[i] = ij_min
            print(f"d is {d}")
            # The if, else code till here is just filling the dataframe as the two points/clusters combine.
            # So, for example if 1 and 2 combines, dataframe will have 1 : 0, 2 : 0. Which means point 1 and 2 both are in same cluster (0th cluster)
            for j in range(0, ij_min[0]):
                # we want to ignore the diagonal, and diagonal is 0. We replaced 0 by infinte.
                # So this if condition will skip diagonals
                if np.isfinite(distance_matrix[ij_min[0]][j]) and np.isfinite(distance_matrix[ij_min[1]][j]):
                    # after two points/cluster are linked, to calculate new distance we take minimum distance for single linkage
                    distance_matrix[ij_min[1]][j] = min(distance_matrix[ij_min[0]][j], distance_matrix[ij_min[1]][j])
            # To avoid the combined data points/cluster in further calculations, we make them infinte.
            # Our if loop above this, will therefore skip the infinite record entries.
            distance_matrix[ij_min[0]] = np.inf

            # print out the information we need
            print(f"Combine datapoint {dataset[d[i][0]]} and datapoint {dataset[d[i][1]]}")


        return d, df[0].to_numpy()

    elif link_type == "complete":
        d_complete = {}
        for i in range(0, numCluster):
            ij_min = np.unravel_index(distance_matrix.argmin(), distance_matrix.shape)
            if i == 0:
                df.iloc[ij_min[0]] = 0
                df.iloc[ij_min[1]] = 0
            else:
                try:
                    a = int(df.iloc[ij_min[0]])
                except:
                    df.iloc[ij_min[0]] = i
                    a = i
                try:
                    b = int(df.iloc[ij_min[1]])
                except:
                    df.iloc[ij_min[1]] = i
                    b = i
                df[(df[0] == a) | (df[0] == b)] = i
            d_complete[i] = ij_min
            for j in range(0, ij_min[0]):
                if np.isfinite(distance_matrix[ij_min[0]][j]) and np.isfinite(distance_matrix[ij_min[1]][j]):
                    # after two points/cluster are linked, to calculate new distance we take maximum distance for complete linkage
                    distance_matrix[ij_min[1]][j] = max(distance_matrix[ij_min[0]][j], distance_matrix[ij_min[1]][j])
            distance_matrix[ij_min[0]] = np.inf

            print(f"Combine datapoint {dataset[d_complete[i][0]]} and datapoint {dataset[d_complete[i][1]]}")

        return d_complete, df[0].to_numpy()


    elif link_type == "average":
        d_average = {}
        for i in range(0, numCluster):
            ij_min = np.unravel_index(distance_matrix.argmin(), distance_matrix.shape)
            if i == 0:
                df.iloc[ij_min[0]] = 0
                df.iloc[ij_min[1]] = 0
            else:
                try:
                    a = int(df.iloc[ij_min[0]])
                except:
                    df.iloc[ij_min[0]] = i
                    a = i
                try:
                    b = int(df.iloc[ij_min[1]])
                except:
                    df.iloc[ij_min[1]] = i
                    b = i
                df[(df[0] == a) | (df[0] == b)] = i
            d_average[i] = ij_min
            for j in range(0, ij_min[0]):
                if np.isfinite(distance_matrix[ij_min[0]][j]) and np.isfinite(distance_matrix[ij_min[1]][j]):
                    # after two points/cluster are linked, to calculate new distance we take average distance for average linkage
                    distance_matrix[ij_min[1]][j] = (distance_matrix[ij_min[0]][j] + distance_matrix[ij_min[1]][
                        j]) / 2.0
            distance_matrix[ij_min[0]] = np.inf

            print(f"Combine datapoint {dataset[d_average[i][0]]} and datapoint {dataset[d_average[i][1]]}")

        return d_average, df[0].to_numpy()

def euclidean_distance(cluster, datapoint):
    """
    For 2 or more dimensions
    :param cluster:
    :param datapoint:
    :return:
    """
    result = []
    for i in range(len(cluster)):
        result.append(np.linalg.norm(cluster[i] - datapoint))
    return result

def abs_distance(cluster, datapoint):
    """
    For 1 dimension
    :param cluster:
    :param datapoint:
    :return:
    """
    result = []
    for i in range(len(cluster)):
        result.append(np.linalg.norm(cluster[i] - datapoint, ord=1))
    return result

# EXECUTION #############################################################

# for k_means
# Note that each "row" is a sample, c is the number of clusters
# cluster_point is the initial clusters
# datapoint = np.asarray([
#     [-1, 3],
#     [1, 4],
#     [0, 5],
#     [4, -1],
#     [3, 0],
#     [5, 1]
# ])
# cluster_point=np.asarray([
#     [-1, 3],
#     [5, 1]
# ])
# k_means(datapoint=datapoint, c=2, cluster_point=cluster_point, randomized=False, mode="euclidean")

# PCA different from that of week7
# new_samples_to_be_classified need to be subtracted by the mean manually first!!
# S = np.asarray([
#     [4,2,2],
#     [0,-2,2],
#     [2,4,2],
#     [-2,0,2]
# ])
# new_samples_to_be_classified = np.asarray([
#         [3,-2,5]
# ])
#
# print(PCA(S,dimension=2,new_samples_to_be_classified=new_samples_to_be_classified))


# competitive_learning_algorithm
# Note that each integer in chosen_order is >= 0
# S = np.asarray([
#     [-1,3],
#     [1,4],
#     [0,5],
#     [4,-1],
#     [3,0],
#     [5,1]
# ], dtype=np.float32)
# initial_centers=np.asarray([
#     [-0.5,1.5],
#     [0,2.5],
#     [1.5,0]
# ])
# chosen_order=np.asarray([2,0,0,4,5])
# new_data_to_be_classfied = np.asarray([
#     [0,-2],
#     [-0.2,2.8],
#     [0,1.5]
# ])
# competitive_learning_algorithm(S,iterations=5,initial_centers= initial_centers,
#                                chosen_order=chosen_order,
#                                new_data_to_be_classfied=new_data_to_be_classfied,
#                                learning_rate=0.1,
#                                normalization_flag=False
#                                )


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
# basic_leader_follower_algorithm(S,iterations=5,initial_centers=initial_centers,
#                                 chosen_order=chosen_order,
#                                 new_data_to_be_classfied=new_data_to_be_classfied,
#                                 learning_rate=0.5,
#                                 theta=3.0,
#                                 normalization_flag=False
#                                 )

# for fuzzyKMeans
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
# fuzzyKMeans(dataset=dataset, numCluster=2, initial_membership=u, b=2, criteria=0.5)

# for agglomerative_clustering
hdataset = np.asarray([
    [-1, 3],
    [1, 2],
    [0, 1],
    [4, 0],
    [5, 4],
    [3, 2]
])
# d , df = agglomerative_clustering(dataset= hdataset, numCluster= 3, link_type="single")
# print("=====Return value ======")
# print(d)
# print(df)


# print(euclidean_distance(np.asarray([[-2.8284, 0],[2.8284, 0]]), np.asarray([-0.7071, -3.5355])))