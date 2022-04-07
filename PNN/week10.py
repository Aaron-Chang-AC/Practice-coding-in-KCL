import numpy as np
import random

def k_means(datapoint, c, randomized=False):
    data = datapoint.copy()
    num_cluster = c
    clusters = []
    if randomized == True: #感覺用不上 但先寫來放著
        for i in range(num_cluster):
            clusters.append(random.choice(data))
            data.remove(clusters[i])
    else:
        # define own m1 and m2 in the list if not randomized
        clusters = np.asarray([[-1, 3], [5, 1]])

    assigned_class_list = []
    assigned_new_class_list = []
    # update this shit
    new_coordinate_1 = []
    new_coordinate_2 = []
    euclidean = []
    round = 1
    while True:
        for i in range(len(datapoint)):
            for j in range(len(clusters)):
                result = np.linalg.norm(datapoint[i] - clusters[j])
                euclidean.append(result)
                # print(result)
            print(euclidean)
            print(f"Argmin is {np.argmin(euclidean)}")
            assigned_new_class_list.append(np.argmin(euclidean))
            print(assigned_new_class_list)
            euclidean[:] = [] # empty list

        if round==1:
            assigned_class_list = assigned_new_class_list.copy()
            # need to improve to make it robustive
            for j in range(len(assigned_class_list)):
                if assigned_class_list[j] == 0:
                    new_coordinate_1.append(datapoint[j])

                elif assigned_class_list[j] == 1:
                    new_coordinate_2.append(datapoint[j])


            m1 =  [sum(sub_list)/ len(sub_list) for sub_list in zip(*new_coordinate_1)]
            m2 = [sum(sub_list)/ len(sub_list) for sub_list in zip(*new_coordinate_2)]
            clusters = [m1, m2]

        else:
            assigned_class_list = assigned_new_class_list.copy()
            if assigned_class_list == assigned_new_class_list:
                print(f"Proccess Done in Round {round}")
                break
            else:
                assigned_new_class_list[:] = []
                for j in range(len(assigned_class_list)):
                    if assigned_class_list[j] == 0:
                        new_coordinate_1.append(datapoint[j])

                    elif assigned_class_list[j] == 1:
                        new_coordinate_2.append(datapoint[j])

                m1 = [sum(sub_list) / len(sub_list) for sub_list in zip(*new_coordinate_1)]
                m2 = [sum(sub_list) / len(sub_list) for sub_list in zip(*new_coordinate_2)]
                clusters = [m1, m2]
        round += 1

    return clusters





datapoint = np.asarray([[-1, 3],[1, 4],[0, 5], [4, -1],[3, 0], [5, 1]])

print(k_means(datapoint, 2, randomized=False))