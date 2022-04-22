import numpy as np
from numpy import linalg as la
import math

def get_profit_from_input(target_number,input):
    for i in input:
        if i[0]==target_number:
            return i[1]
def get_weight_from_input(target_number,input):
    for i in input:
        if i[0]==target_number:
            return i[2]
def get_profit_from_solution(solution,input):
    temp=0.0
    for i in input:
        if i[0] in solution:
            temp+=i[1]
    return temp
def get_weight_from_solution(solution,input):
    temp=0.0
    for i in input:
        if i[0] in solution:
            temp+=i[2]
    return temp
def find_solution_index(solution_dict, solution):
    for i in range(len(solution_dict)):
        if solution == solution_dict[i]:
            return i
def check_set_in_dict(s,D):
    flag = False
    for i in range(len(D)):
        if s == D[i]:
            flag = True
            break
    return flag
# def calculate_branch_and_upperbound(item_set)
def greedy(input,solution,maximum_weight):
    current_profits=0.0
    current_weights=0.0
    for i in solution:
        for j in input:
            if j[0]==i:
                current_profits+=j[1]
                current_weights+=j[2]
    temp = {}
    for i in input:
        if int(i[0]) not in solution:
            temp[int(i[0])] = i[1] / i[2]

    max_key = max(temp, key=temp.get)
    # print(current_profits,max_key)
    temp = {k: v for k, v in sorted(temp.items(), key=lambda item: item[1],reverse=True)}
    weight_temp=current_weights
    profit_temp=current_profits
    for i in temp.keys():
        wi=get_weight_from_input(i,input)
        pi=get_profit_from_input(i,input)
        if (weight_temp+wi) <= maximum_weight:
            weight_temp += wi
            profit_temp += pi
        else:
            profit_temp += ((maximum_weight-weight_temp)/wi*pi)
            weight_temp = maximum_weight
            break
    # print(profit_temp)
    return profit_temp

def extract_max(input,solution_dict,maximum_weight):
    max_upper_bound=0.0
    max_solution=set([])
    for i in range(len(solution_dict)):
        profit_temp = greedy(input, solution_dict[i], maximum_weight)
        if profit_temp>max_upper_bound:
            max_upper_bound=profit_temp
            max_solution=solution_dict[i].copy()
    return max_solution, max_upper_bound

def branch_bound_knapsack_problem(input,maximum_weight):
    solution_dict={}
    visited_solution_dict={}
    printed_solution_dict={}
    cnt=len(solution_dict)
    cnt_v=len(visited_solution_dict)
    cnt_p=len(printed_solution_dict)

    
    initial_item_list=[]
    initial_item_set={}
    for i in range(len(input)):
        weight_i = get_weight_from_solution(set([int(input[i,0])]), input)
        if weight_i<=maximum_weight:
            solution_dict[cnt] = set([int(input[i, 0])])
            cnt += 1
        initial_item_list.append(int(input[i,0]))
        visited_solution_dict[cnt_v]=set([int(input[i,0])])
        cnt_v+=1
    all_item_set=set(initial_item_list)

    print(f"\ninput(index,profit,weight):\n{input}")
    print(f"\ninitial_item_set:\n{all_item_set}")
    print(f"\ninitial_solution_dict:\n{solution_dict}")
    max = -np.Inf
    true_max_solution = set([])
    true_max_solution_upperbound = 0 # roger
    true_max_solution_weight = 0 # roger
    while len(solution_dict) > 0:
        print(f"\n-----------------current_solution_dict:-----------------\n{solution_dict}")
        max_solution, max_upper_bound = extract_max(input, solution_dict, maximum_weight)
        print(f"\nmax_solution in current solution_dict:\n{max_solution}\n")
        print(f"\nupper_bound of max_solution:\n{max_upper_bound}\n")
        solution_weight = get_weight_from_solution(max_solution, input)
        solution_profit = get_profit_from_solution(max_solution, input)

        if solution_profit >= max:
            true_max_solution = max_solution.copy()
            max = solution_profit
            true_max_solution_upperbound = max_upper_bound # roger
            true_max_solution_weight = solution_weight # roger
        
        if max_upper_bound<max:
            break

        temp = max_solution ^ all_item_set
        temp_candidate_solution={}
        temp_candidate_solution_cnt=0
        # print(temp)
        for i in temp:
            temp_candidate_solution[temp_candidate_solution_cnt]=set(max_solution|set([i]))
            temp_candidate_solution_cnt+=1
        print(f"\nsolutions generated after bound:\n")
        for i in range(len(temp_candidate_solution)):
            weight_i = get_weight_from_solution(temp_candidate_solution[i],input)
            profit_i = get_profit_from_solution(temp_candidate_solution[i],input)
            if (len(temp_candidate_solution[i])<len(input)) and (weight_i<maximum_weight) and not(check_set_in_dict(temp_candidate_solution[i],visited_solution_dict)):
                solution_dict[cnt] = temp_candidate_solution[i].copy()
                visited_solution_dict[cnt_v] = temp_candidate_solution[i].copy()
                cnt += 1
                cnt_v += 1
            if (len(temp_candidate_solution[i])<=len(input)) and (weight_i<=maximum_weight) and not(check_set_in_dict(temp_candidate_solution[i],printed_solution_dict)):
                print(f"solution: {temp_candidate_solution[i]} , profit: {profit_i} , weight: {weight_i}\n")
                printed_solution_dict[cnt_p]=temp_candidate_solution[i].copy()
                cnt_p+=1
                if profit_i>=max:
                    max=profit_i
                    true_max_solution=temp_candidate_solution[i].copy()
                    true_max_solution_upperbound = max_upper_bound # roger
                    true_max_solution_weight = weight_i # roger
            elif not check_set_in_dict(temp_candidate_solution[i],printed_solution_dict):
                print(f"invalid solution!!: {temp_candidate_solution[i]} , profit: {profit_i} , weight: {weight_i}\n")
                printed_solution_dict[cnt_p] = temp_candidate_solution[i].copy()
                cnt_p += 1
        index = find_solution_index(solution_dict, max_solution)
        # print(solution_dict, max_solution)
        del solution_dict[index]
        cnt-=1
        new_solution_dict={}
        for i in solution_dict.keys():
            k = len(new_solution_dict)
            new_solution_dict[k]=solution_dict[i]
        solution_dict = new_solution_dict

    print("============Final Result=============")
    print(f"true_max_solution:\n{true_max_solution}")
    print(f"maximum profit: {max}")
    print(f"true_max_solution_weight: {true_max_solution_weight}") #roger
    print(f"true_max_solution_upper_bound: {true_max_solution_upperbound}") # roger
    return
    
# input = np.asarray([
#     [1,4,6],
#     [2,5,4],
#     [3,6,3],
#     [4,5,10]
# ], dtype=np.float32)
input = np.asarray([
    [1, 4, 6],
    [2, 5, 4],
    [3, 6, 3],
    [4, 5, 10]
], dtype=np.float32)
branch_bound_knapsack_problem(input,maximum_weight=15.0)