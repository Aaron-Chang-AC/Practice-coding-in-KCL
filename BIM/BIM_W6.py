import numpy as np
from numpy import linalg as la
import math
import pandas as pd
import itertools

def read_graph_from_csv():
    df = pd.read_csv('graph.csv', encoding='unicode_escape')
    return df

def calculate_cut_cost(L, R, df_bidirectional):
    cost=0.0
    for i in range(len(L)):
        for j in range(len(R)):
            temp = df_bidirectional.loc[
                (df_bidirectional['node1'] == L[i]) & (
                    df_bidirectional['node2'] == R[j])].to_numpy()
            # print(temp)
            if len(temp)>0:
                cost += temp[0, 2]
    return cost

def weighted_graph_bisection_problem_SA(L=None, R=None, random_numbers=None, initial_temperature=None, temperature_reducing_rate=None, k=None, transitions_each_phase=None):
    df = read_graph_from_csv().drop_duplicates(ignore_index=True)
    num_nodes = df[['node1', 'node2']].max().max() - df[['node1', 'node2']].min().min()+ 1
    print(f"Graph:\n{df}")
    print(f"number of nodes: {num_nodes}\nnumber of edges: {len(df)}")
    edge_temp = df[['node1', 'node2']].to_numpy()
    EDGES = np.append(edge_temp, df[['node2', 'node1']].to_numpy(), axis=0)
    weights_temp = df['weight'].to_numpy()
    weights = np.append(weights_temp, df['weight'].to_numpy())
    df_bidirectional = pd.DataFrame({'node1': EDGES[:, 0], 'node2': EDGES[:, 1], 'weight': weights})
    df_bidirectional = df_bidirectional.drop_duplicates(ignore_index=True)
    # print(df_bidirectional)
    random_numbers_index=0
    current_cost = calculate_cut_cost(L, R, df_bidirectional)
    current_temperature=initial_temperature
    print(f"initail condition:\nL:{L}\nR:{R}")
    optimal_L = L.copy()
    optimal_R = R.copy()
    optimal_cost = current_cost

    for i in range(k):
        print(f"----------------phase {i+1}----------------")
        for j in range(transitions_each_phase):
            rand_1 = random_numbers[random_numbers_index]
            rand_2 = random_numbers[random_numbers_index + 1]
            print(f"rand_1:{rand_1}, rand_2:{rand_2}")
            swap_index_1 = 0
            swap_index_2 = 0
            random_numbers_index += 2
            for u in range(len(L)):
                if rand_1 < ((1.0/len(L))*(u+1)):
                    swap_index_1 = u
                    break
            for v in range(len(R)):
                if rand_2 < ((1.0/len(R))*(v+1)):
                    swap_index_2 = v
                    break
            new_L = L.copy()
            new_R = R.copy()
            new_L[swap_index_1] = R[swap_index_2]
            new_R[swap_index_2] = L[swap_index_1]
            temp_cost = calculate_cut_cost(new_L, new_R, df_bidirectional)
            print(f"current_cost:{current_cost}, new_cost:{temp_cost}")
            print(f"found new condition:\nnew_L:{new_L}\nnew_R:{new_R}\n\n")
            if temp_cost<current_cost:
                print(f"accept new condition: new_cost < current_cost")
                current_cost=temp_cost
                L = new_L.copy()
                R = new_R.copy()
                if temp_cost<optimal_cost:
                    optimal_L = L.copy()
                    optimal_R = R.copy()
                    optimal_cost = temp_cost
            else:
                current_random_number = random_numbers[random_numbers_index]
                random_numbers_index += 1
                acceptance_prob = math.exp((current_cost - temp_cost) / current_temperature)
                print(f"current random number: {current_random_number}")
                print(f"current temperature: {current_temperature}\nacceptance prob: {acceptance_prob}")
                if current_random_number < acceptance_prob:
                    current_cost = temp_cost
                    L = new_L.copy()
                    R = new_R.copy()
                    print(f"accept new condition: current_random_number < acceptance_prob")
                    if temp_cost < optimal_cost:
                        optimal_L = L.copy()
                        optimal_R = R.copy()
                        optimal_cost = temp_cost
            print(f"final condition in this iteration:\nL:{L}\nR:{R}\n\n")
            current_temperature=temperature_reducing_rate*current_temperature

    print(f"optimal solution in weighted_graph_bisection_problem_SA:")
    print(f"optimal_L:{optimal_L}\noptimal_R:{optimal_R}")
    print(f"optimal cost: {optimal_cost}")
    
    return
    

# EXECUTION_________________________
weighted_graph_bisection_problem_SA(L=np.asarray([3,4,6,1,10],dtype=np.int8), R=np.asarray([2,5,7,8,9],dtype=np.int8),
                                    random_numbers=np.asarray([0.7, 0.56, 0.16, 0.35, 0.32, 0.45, 0.67, 0.12, 0.78],dtype=np.float32),
                                    initial_temperature=100, temperature_reducing_rate=0.9, k=2, transitions_each_phase=1)
