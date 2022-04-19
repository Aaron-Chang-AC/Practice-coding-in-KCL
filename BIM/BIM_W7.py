import numpy as np
from numpy import linalg as la
import math
import pandas as pd
import itertools

def roulette_wheel_selection(individuals_fitness_dict=None):
    print(f"applying roulette_wheel_selection")
    total_fitness=float(sum(individuals_fitness_dict.values()))
    print(f"total fitness: {total_fitness}")
    for i in individuals_fitness_dict.keys():
        print(f"individual {i}: selection prob. = {round(individuals_fitness_dict[i]/total_fitness,5)}")
    print(f"\n\n")
    return

def rank_selection(individuals_fitness_dict=None):
    print(f"applying rank_selection")
    keys = individuals_fitness_dict.keys()
    individuals_fitness_dict = dict(sorted(individuals_fitness_dict.items(), key=lambda item: item[1]))
    val=1
    for i in individuals_fitness_dict.keys():
        individuals_fitness_dict[i]=val
        val+=1
    total_rank = float(sum(individuals_fitness_dict.values()))
    print(f"total rank: {total_rank}")
    for i in keys:
        print(f"individual {i}: selection prob. = {round(individuals_fitness_dict[i] / total_rank, 5)}")
    print(f"\n\n")
    return

def tournament_selection(individuals_fitness_dict=None, pick_k_individuals=None):
    print(f"applying tournament_selection")
    keys = individuals_fitness_dict.keys()
    prob_dict={}
    for i in keys:
        prob_dict[i]=0.0
    comb = list(itertools.combinations(keys, pick_k_individuals))
    total_combinations = len(comb)
    print(f"total combinations: {total_combinations}")
    for i in comb:
        temp={}
        for j in range(pick_k_individuals):
            temp[i[j]]=individuals_fitness_dict[i[j]]
        max_key = max(temp, key = temp.get)
        prob_dict[max_key] += 1
    for i in keys:
        prob_dict[i] /= total_combinations
        print(f"individual {i}: selection prob. = {round(prob_dict[i],5)}")
    print(f"\n\n")
    return

# EXECUTION_________________________
'''
# individuals_fitness_dict: key -> names of individuals, values -> fitness of individuals
individuals_fitness_dict = {
    "a":3,
    "b":9,
    "c":1,
    "d":7,
    "g":12,
    "h":8
}
roulette_wheel_selection(individuals_fitness_dict=individuals_fitness_dict)
rank_selection(individuals_fitness_dict=individuals_fitness_dict)
tournament_selection(individuals_fitness_dict=individuals_fitness_dict, pick_k_individuals=3)
'''