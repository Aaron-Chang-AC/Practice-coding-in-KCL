import numpy as np
import random
from scipy.spatial.distance import cityblock

import pandas as pd
import copy

class Cluster:
    def __init__(self):
        self.samples = {}
        self.num_samples = 0
    def add_sample(self,sample):
        self.samples[self.num_samples] = sample.copy()
        self.num_samples+=1
    def get_samples(self):
        result = []
        for i in range(self.num_samples):
            result.append(self.samples[i])
        return result
    def show_cluster(self):
        for i in range(self.num_samples):
            print(f"sample{i+1}: {self.samples[i]}")

def calculate_cluster_distance(c1,c2,link_type,ord):
    dist = np.Inf
    samples1 = np.asarray(c1.get_samples())
    samples2 = np.asarray(c2.get_samples())
    if link_type == "single":
        for i in range(len(samples1)):
            for j in range(len(samples2)):
                temp = np.linalg.norm(samples1[i] - samples2[j],ord=ord)
                if temp < dist:
                    dist = temp
    elif link_type == "complete":
        dist = -np.Inf
        for i in range(len(samples1)):
            for j in range(len(samples2)):
                temp = np.linalg.norm(samples1[i] - samples2[j],ord=ord)
                if temp > dist:
                    dist = temp
    elif link_type == "average":
        dist = 0.0
        for i in range(len(samples1)):
            for j in range(len(samples2)):
                dist += np.linalg.norm(samples1[i] - samples2[j],ord=ord)
        dist = dist / (len(samples1)*len(samples2))
    else:
        dist = 0.0
        mean1 = np.mean(samples1, axis=0)
        mean2 = np.mean(samples2, axis=0)
        dist = np.linalg.norm(mean1 - mean2,ord=ord)
    return dist

def Agglomerative_clustering(dataset, numCluster, link_type=None,ord=2):
    clusters = {}
    n = len(dataset)
    epoch_cnt = 0
    for i in range(n):
        print(f"cluster {i+1}")
        clusters[i] = Cluster()
        clusters[i].add_sample(dataset[i])
        clusters[i].show_cluster()
        print("\n")
    while len(clusters) > numCluster:
        epoch_cnt+=1
        print(f"------------EPOCH {epoch_cnt} -----------")
        m = len(clusters)
        dist = np.zeros((m, m), dtype=np.float32)
        minimum = np.Inf
        min_idx_1 = 0
        min_idx_2 = 0
        for i in range(m):
            for j in range(m):
                dist[i, j] = np.Inf
                if (i > j):
                    dist[i,j]=calculate_cluster_distance(clusters[i], clusters[j], link_type, ord)
                    if dist[i,j] < minimum:
                        minimum = dist[i,j]
                        idx = sorted([i,j])
                        min_idx_1 = idx[0]
                        min_idx_2 = idx[1]
        print(f"distances:\n{dist}\n")
        temp = clusters[min_idx_2].get_samples()
        for i in range(len(temp)):
            clusters[min_idx_1].add_sample(temp[i])
        print(f"minimum: {minimum}")
        print(f"merge cluster {min_idx_1+1} and {min_idx_2+1}")
        del clusters[min_idx_2]
        new_clusters = {}
        cnt = 0
        for i in clusters.keys():
            new_clusters[cnt] = copy.deepcopy(clusters[i])
            cnt+=1
        clusters = copy.deepcopy(new_clusters)
        print("\n")
        for i in clusters.keys():
            print(f"cluster {i+1}")
            clusters[i].show_cluster()
            print("\n")
    print(f"------------Final Clusters -----------")
    for i in clusters.keys():
        print(f"cluster {i + 1}")
        clusters[i].show_cluster()
        print("\n")
    return

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



from collections import Counter
def iterative_optimization(datapoint, initial_clustering, clusters, selected_datapoint_tomove=None, maximum_iteration=1):
    """
    Basic idea lies in starting from a reasonable initial partition and move samples from one cluster
    to another trying to minimize criterion function

    :param datapoint:
    :return:
    """
    num_samples = len(datapoint) # n
    num_cluster = len(clusters) # c
    # number of points in cluster
    n = dict()
    # total cost dictionary
    J = dict()
    # initialize z first
    for j in range(num_cluster):
        J[j] = 0
    # find initial loss
    for i in range(num_samples):
        for j in range(num_cluster):
            if initial_clustering[i] == j:
                J[j] += np.linalg.norm(datapoint[i] - clusters[j])
    print(f"Initial total loss: {J}")
    cnt = 1
    initial_clustering_copy = initial_clustering.copy()


    print(f"Initial clustering is: {initial_clustering}")
    assigned_cluster = []
    for i in range(num_samples):
        print(f"==========Round {cnt}--Sample {i+1}: {datapoint[i]}============")
        init = []
        for j in range(num_cluster):
            result = np.linalg.norm(datapoint[i] - clusters[j])
            init.append(result)
        # print(init)
        assigned_cluster.append(np.argmin(init))
        initial_clustering[i] = np.argmin(init)
        print(f"Initial assigned cluster before ro {initial_clustering}")
        print(f"i<-- {assigned_cluster}")
        n = Counter(initial_clustering)
        print(f"Count of number of datapoint in each cluster:{n}")
    # print(f"Assigned Cluster: {assigned_cluster}")

        # initialize ro_list and loss
        ro_list = {}
        for j in range(num_cluster):
            ro_list[j] = 0

        print(f"ro_list is initialized: {ro_list}")

        if len(assigned_cluster) > 0: # not destroying singleton cluster
            old_loss = J.copy() # current loss for comparision
            print(f"Old loss is: {old_loss}")
            for j in range(num_cluster):
                if assigned_cluster[i] == j:
                    try:
                        ro = (n[j]/(n[j]-1)) * (np.linalg.norm(datapoint[i]-clusters[j]))
                    except ZeroDivisionError:
                        ro = 0
                    print(f"when i==j: {ro}")
                    ro_list[j] = ro
                    J[j] -= ro
                elif assigned_cluster[i] != j:
                    try:
                        ro = (n[j]/(n[j]+1)) * (np.linalg.norm(datapoint[i]-clusters[j]))
                    except ZeroDivisionError:
                        ro = 0
                    print(f"when i!=j: {ro}")
                    ro_list[j] = ro
                    J[j] += ro
            print(f"ro_j in each cluster is :{ro_list}")
            k = min(ro_list, key=ro_list.get)
            if initial_clustering[i] == k:
                print("Original cluster is better")
            else:
                print("Change to another cluster")
                # update the assigned cluster
                assigned_cluster[i] = k
                initial_clustering[i] = k
                print(f"New assigned cluster after ro:{initial_clustering}")


            # recompute cluster point
            datapoint_classify_dict = {}
            for j in range(num_cluster):
                datapoint_classify_dict[j] = []
            for k in range(num_samples):
                for j in range(num_cluster):
                    if initial_clustering[k] == j:
                        datapoint_classify_dict[j].append([datapoint[k]])

            # update cluster point
            for keys in datapoint_classify_dict:
                for j in range(num_cluster):
                    if keys == j:
                        clusters[j] = np.mean(datapoint_classify_dict[keys], axis=0)


            print(f"Updated New clusters are: \n {clusters}")
            # recompute J
            print(J)
            print(f"Sum of the total loss is: {sum(J.values())}")


            if sum(J.values()) < sum(old_loss.values()):
                print("###############J is smaller now###############")
                print("==============================================")
                print(f"Final cluster is:\n {clusters}")
                print(f"Initial classified list: \n {initial_clustering_copy}")
                print(f"Final classified list: \n {initial_clustering}")
                return clusters
            else:
                print("J is still larger, continue")



    print("==============================================")
    print(f"Final cluster is:\n {clusters}")
    print(f"Initial classified list: \n {initial_clustering_copy}")
    print(f"Final classified list: \n {initial_clustering}")
    return clusters

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

# iterative_optimization(datapoint=datapoint, initial_clustering= initial_clustering, clusters=cluster_point, selected_datapoint_tomove= selected_datapoint_tomove)

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

# # for agglomerative_clustering
# hdataset = [
#     [-1, 3],
#     [1, 2],
#     [0, 1],
#     [4, 0],
#     [5, 4],
#     [3, 2]
# ]
# Agglomerative_clustering(dataset= hdataset, numCluster= 3, link_type="single", ord=2)
# Agglomerative_clustering(dataset= hdataset, numCluster= 3, link_type="complete", ord=2)
# Agglomerative_clustering(dataset= hdataset, numCluster= 3, link_type="average", ord=2)
# Agglomerative_clustering(dataset= hdataset, numCluster= 3, link_type="mean", ord=2)

# print(euclidean_distance(np.asarray([[-2.8284, 0],[2.8284, 0]]), np.asarray([-0.7071, -3.5355])))