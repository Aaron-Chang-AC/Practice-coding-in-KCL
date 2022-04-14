import numpy as np
from numpy import linalg as la
import math
import pandas as pd
import itertools


def read_graph_from_csv():
    df = pd.read_csv('graph.csv', encoding='unicode_escape')
    return df

# find primes below an upper_bound
def find_prime(upper_bound):
    print(f"finding primes in range({upper_bound})")
    primes=[]
    for num in range(1, upper_bound + 1):
        if num > 1:
            for i in range(2, num):
                if (num % i) == 0:
                    break
            else:
                primes.append(num)
    print("number of primes: ",len(primes))
    print("Primes: ",primes)
    return primes
# number of primes bad for x and y
def bad_for_xy(x,y,primes):
    cnt=0.0
    primes_bad_for_x_y=[]
    for i in primes:
        if (x%i) == (y%i):
            primes_bad_for_x_y.append(i)
            cnt+=1
    print(f"primes bad for x and y: {primes_bad_for_x_y}")
    print(f"number of primes bad for x and y: {int(cnt)}")
    return cnt

# calculate error of witness
def error_witness(number_of_bits,x,y):
    print(f"the number of bits (size) of x and y: {number_of_bits}")
    print(f"x = {x}")
    print(f"y = {y}")
    primes=find_prime(number_of_bits**2)
    cnt=bad_for_xy(x,y,primes)
    err = cnt/len(primes)
    print(f"Error of witness: {err}")

def TSP():
    df = read_graph_from_csv()
    solution_dict={}
    cnt=0
    num_nodes = df[['node1','node2']].max().max()+1
    print(f"Graph:\n{df}")
    print(f"number of nodes: {num_nodes}")
    index_list = np.asarray(range(num_nodes))
    permutation_arr = np.asarray(list(itertools.permutations(index_list[1:])))
    # if i_flip is in solution_dict -> True
    flag = False
    for i in permutation_arr:
        if cnt == 0:
            solution_dict[cnt] = i.copy()
            cnt += 1
            continue
        for j in solution_dict.keys():
            if np.array_equal(np.flip(i), solution_dict[j]):
                flag = True
                break

        if flag:
            flag = False
            continue
        else:
            solution_dict[cnt] = i.copy()
            cnt += 1
    for i in solution_dict.keys():
        temp = np.zeros(num_nodes+1,dtype=np.int8)
        temp[1:num_nodes] = solution_dict[i].copy()
        solution_dict[i] = temp

    print(f"number of tours: {len(solution_dict)}")
    edge_temp = df[['node1','node2']].to_numpy()
    EDGES = np.append(edge_temp,df[['node2','node1']].to_numpy(), axis=0)
    weights_temp = df['weight'].to_numpy()
    weights = np.append(weights_temp,df['weight'].to_numpy())
    df_bidirectional = pd.DataFrame({'node1': EDGES[:,0], 'node2': EDGES[:,1], 'weight': weights})
    
    minimum_cost=np.Inf
    solution_weight_sum = []
    opt_solution_cnt = 0
    for i in solution_dict.keys():
        weight=0.0
        for j in range(num_nodes):
            node1 = solution_dict[i][j]
            node2 = solution_dict[i][j+1]
            temp = df_bidirectional.loc[(df_bidirectional['node1'] == node1) & (df_bidirectional['node2'] == node2)].to_numpy()
            weight += temp[0,2]
        solution_weight_sum.append(weight)
        if weight<=minimum_cost:
            minimum_cost=weight
        print(f"soltion {i+1}: {solution_dict[i].tolist()}, total_weight: {weight}")
    print(f"minimum cost = {minimum_cost}")
    for i in solution_dict.keys():
        if math.isclose(solution_weight_sum[i] , minimum_cost):
            opt_solution_cnt += 1
            print(f"optimal solution {opt_solution_cnt}: {solution_dict[i]}")
    print(f"number of optimal solutions: {opt_solution_cnt}")
    print(f"probability of finding an optimum tour when generating a tour at random: {float(opt_solution_cnt)/len(solution_dict)}")

TSP()
'''
for week4 tutorial Q1

# you can enter decimal number like this:
error_witness(number_of_bits=8,x=54,y=173)

# or you can enter binary number like this:
error_witness(number_of_bits=8,x=int('00011011',2),y=int('10101101',2))
'''

'''
for week4 tutorial Q3 : Note that modifing the csv file is needed when the question changes

TSP()

'''