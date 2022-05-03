import numpy as np
from dimacs_parser import parse
import random
import time


class WalkSAT_Solver:

    def __init__(self, input_cnf_file, initial_assignment, verbose):
        self.list_clauses, self.nvars = parse(input_cnf_file, verbose)
        self.verbose = verbose # 0 or 1
        self.inital_assignment = initial_assignment # roger
        self.assignment = []
        self.pool = dict()  # key: literal -> element: index of clause which contains literal
        self.unsat_clauses = []  # save id of unsat clause
        self.costs = np.zeros(
            len(self.list_clauses))  # compute nb of literals make clause true (i.e. for clause Ci, if fi>0 => T, fi==0 => F)
        self.MAX_TRIES = 100
        self.MAX_FLIPS = 500
        self.nb_tries = 0
        self.nb_flips = 0
        self.is_sat = False
        self.noise_parameter = 0.2

        for clause in self.list_clauses:
            assert len(clause) > 0

    def generate(self):
        if self.inital_assignment:
            self.assignment = self.inital_assignment
        else:
            self.assignment = []
            self.nb_tries += 1
            self.nb_flips = 0
            for x in range(1, self.nvars + 1):
                choice = [-1, 1]
                self.assignment.append(x * random.choice(choice))

    def initialize_pool(self):
        for i, clause in enumerate(self.list_clauses):
            for literal in clause:
                if literal in self.pool.keys():
                    self.pool[literal].append(i)
                else:
                    self.pool[literal] = [i]

    def initialize_cost(self):
        # Compute nb of literals make clause true (i.e. for clause Ci, if fi>0 => T, fi==0 => F)
        # Let's call it cost !
        assert len(self.assignment) > 0
        self.unsat_clauses = []
        for i, clause in enumerate(self.list_clauses):
            self.costs[i] = 0
            for literal in clause:
                if literal in self.assignment:
                    self.costs[i] += 1
            if self.costs[i] == 0:  # Clause[i] is currently UNSAT
                self.unsat_clauses.append(i)

    def check(self):
        # check if all is SAT
        return len(self.unsat_clauses) == 0

    def pick_unsat_clause(self):
        assert len(self.unsat_clauses) > 0
        random_index = random.choice(self.unsat_clauses)
        return self.list_clauses[random_index]

    def evaluate_breakcount(self, literal, bs=1, ms=1):
        # Compute the breakcount score: #clause which turn SAT -> UNSAT
        if literal in self.assignment:
            original_literal = literal
        elif -literal in self.assignment:
            original_literal = -literal
        # when flipping literal -> -literal
        # For every clause which contains literal => cost--
        breakcount = 0
        if original_literal in self.pool.keys():
            for i in self.pool[original_literal]:
                if self.costs[i] == 1:
                    breakcount += 1
        # For every clause which contains -literal => cost ++
        makecount = 0
        if -original_literal in self.pool.keys():
            for j in self.pool[-original_literal]:
                if self.costs[j] == 0:
                    makecount += 1
        # Score = break - make
        score = bs * breakcount - ms * makecount
        return score

    def flip(self, literal):
        self.nb_flips += 1
        # Flip variable in assignment
        if literal in self.assignment:
            ind = self.assignment.index(literal)
        elif -literal in self.assignment:
            ind = self.assignment.index(-literal)
        old_literal = self.assignment[ind]
        self.assignment[ind] *= -1
        # Update cost
        # Clause contains literal => cost --
        if old_literal in self.pool.keys():
            for i in self.pool[old_literal]:
                self.costs[i] -= 1
                if self.costs[i] == 0:  # if SAT -> UNSAT: add to list of  unsat clauses
                    self.unsat_clauses.append(i)
        # Clause contains -literal => cost ++
        if -old_literal in self.pool.keys():
            for j in self.pool[-old_literal]:
                if self.costs[j] == 0:  # if UNSAT -> SAT: remove from list of unsat clauses
                    self.unsat_clauses.remove(j)
                self.costs[j] += 1

    def solve(self):
        initial = time.time()
        self.initialize_pool()
        iter = 1
        while self.nb_tries < self.MAX_TRIES and not self.is_sat:
            self.generate()
            print(f"Initial configuration is {self.assignment}") # roger
            self.initialize_cost()
            while self.nb_flips < self.MAX_FLIPS and not self.is_sat:
                print(f"===========Iteration {iter}==================")
                if self.check() == 1:  # if no unsat clause => finish
                    self.is_sat = True
                    if iter == 1:
                        print("The original solution is SAT")
                    else:
                        print(f"Process ended in --{iter}-- round")
                else:
                    assert len(self.unsat_clauses) > 0
                    # Choose a variable x to flip
                    unsat_clause = self.pick_unsat_clause()
                    print(f"Unsatisfied Clause: {unsat_clause}") # roger
                    break_count = []
                    for literal in unsat_clause:
                        break_count.append(self.evaluate_breakcount(literal))
                    # if 0 in break_count: # that's an excellent x
                    #     x = unsat_clause[break_count.index(0)]
                    # else:
                    p = random.random()
                    if p < self.noise_parameter:  # pick x randomly from unsat clause
                        x = random.choice(unsat_clause)
                    else:
                        x = unsat_clause[np.argmin(break_count)]
                    old = self.assignment.copy() # roger
                    self.flip(x)
                    print(f"Current configuration: {self.assignment}")
                    difference = list(set(old)^set(self.assignment))
                    print(f"Number flipped from {difference[0]} to {difference[1]}")
                iter += 1

        end = time.time()
        print('Nb flips:  {0}      '.format(self.nb_flips))
        print('Nb tries:  {0}      '.format(self.nb_tries))
        print('CPU time:  {0:10.4f} s '.format(end - initial))
        if self.is_sat:
            print('SAT')
            print(f"Final Satisfied Assignment is: {self.assignment}")
            return self.assignment
        else:
            print('UNKNOWN')
            return None

    def assignment_to_binary(self):
        binary_assignment = []
        for i in self.assignment:
            if i < 0:
                binary_assignment.append(0)
            elif i > 0:
                binary_assignment.append(1)
        print(f"Binary Satisfied Assignment is: {binary_assignment}")
        return binary_assignment

def main():
    # try:
    #     args = get_args()
    #     input_cnf_file = args.input
    #     verbose = args.verbose
    #
    # except:
    #     print("missing or invalid arguments")
    #     exit(0)
    input_cnf_file = "C:/Users/KuanHaoChen/Documents/GitHub/EXAMPREP/BIM/cnf.cnf"
    verbose = 1  # 0 or 1, 1 to see details
    # if assignment is empty, generate randomly
    assignment = [1, -2, -3, -4, -5, -6, 7, 8, -9, 10, 11, 12, -13, -14, 15, 16, 17, -18, -19, -20]
    solver = WalkSAT_Solver(input_cnf_file, assignment, verbose)
    solver.solve()
    solver.assignment_to_binary()

if __name__ == '__main__':
    main()