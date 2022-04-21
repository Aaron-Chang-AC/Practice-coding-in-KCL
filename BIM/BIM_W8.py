import numpy as np
import itertools
import math
import copy

class task:
    def __init__(self, task_name, machine, processing_time):
        self.task_name = task_name
        self.machine = machine
        self.processing_time = processing_time
        self.finished = False
        self.parent = -1
    def set_finished(self):
        self.finished = True
    def set_parent(self, parent_task):
        self.parent = parent_task

class machine:
    def __init__(self):
        self.busy_or_not = False
        self.remain_processing_time = 0
        self.processing_task = 0

    def modify_remain_time(self):
        if self.busy_or_not:
            self.remain_processing_time -= 1
        if math.isclose(self.remain_processing_time, 0):
            self.remain_processing_time = 0
            self.busy_or_not = False

    def assign_task(self, task, time):
        self.processing_task = task
        self.remain_processing_time = time
        self.busy_or_not = True

# from https://stackoverflow.com/questions/9660085/python-permutations-with-constraints
def _permute(L, nexts, numbers, begin, end):
    if end == begin + 1:
        yield L
    else:
        for i in range(begin, end):
            c = L[i]
            if nexts[c][0] == numbers[c]:
                nexts[c][0] += 1
                L[begin], L[i] = L[i], L[begin]
                for p in _permute(L, nexts, numbers, begin + 1, end):
                    yield p
                L[begin], L[i] = L[i], L[begin]
                nexts[c][0] -= 1


def constrained_permutations(L, constraints):
    # warning: assumes that L has unique, hashable elements
    # constraints is a list of constraints, where each constraint is a list of elements which should appear in the permatation in that order
    # warning: constraints may not overlap!
    nexts = dict((a, [0]) for a in L)
    numbers = dict.fromkeys(L, 0) # number of each element in its constraint
    for constraint in constraints:
        for i, pos in enumerate(constraint):
            nexts[pos] = nexts[constraint[0]]
            numbers[pos] = i

    for p in _permute(L, nexts, numbers, 0, len(L)):
        yield p
########################################################################################
def return_task_from_jobs(task_name=None, jobs=None):
    for i in jobs:
        for j in i:
            if j.task_name == task_name:
                return j
def calculate_makespan(solution=None, jobs=None, machines=None):
    makespan=0.0
    machine_names=machines.keys()
    remaining_task = solution.copy()
    for i in jobs:
        for j in range(len(i)):
            if j>0:
                parent_task = i[j-1].task_name
                i[j].set_parent(parent_task)

    while True:
        if len(remaining_task) > 0:
            # assign tasks if corresponding machines are free
            pop_idx=[]
            for i in range(len(remaining_task)):
                temp = remaining_task[i]
                task_temp = return_task_from_jobs(temp, jobs)
                temp_parent = task_temp.parent
                if temp_parent != -1:
                    task_temp_parent = return_task_from_jobs(temp_parent, jobs)
                    if not (machines[task_temp.machine].busy_or_not) and task_temp_parent.finished:
                        machines[task_temp.machine].assign_task(task_temp.task_name, task_temp.processing_time)
                        pop_idx.append(i)
                    else:
                        break
                elif temp_parent == -1:
                    if not(machines[task_temp.machine].busy_or_not):
                        machines[task_temp.machine].assign_task(task_temp.task_name, task_temp.processing_time)
                        pop_idx.append(i)
                    else:
                        break
                else:
                    break
            if len(pop_idx)>0:
                for i in range(len(pop_idx)):
                    temp = remaining_task.pop(0)

        # time moving forward
        busy_machines_after_assignment_cnt=0
        for m in machine_names:
            # print(f"machine {m}: {machines[m].busy_or_not}")
            if machines[m].busy_or_not:
                busy_machines_after_assignment_cnt+=1
        if busy_machines_after_assignment_cnt == 0:
            break
        else:
            makespan += 1
            for m in machine_names:
                machines[m].modify_remain_time()
                if (machines[m].processing_task != 0) and not(machines[m].busy_or_not):
                    task_temp = return_task_from_jobs(machines[m].processing_task, jobs)
                    task_temp.set_finished()
    return makespan


def job_scheduling(jobs=None, machines=None):
    task_list = []
    constraints = []
    constraints_idx = 0
    for i in range(len(jobs)):
        constraints.append([])
        for j in range(len(jobs[i])):
            task_list.append(jobs[i][j].task_name)
            constraints[constraints_idx].append(jobs[i][j].task_name)
        constraints_idx+=1
    task_num = len(task_list)
    print(f"task_list: {task_list}\nnumber of tasks: {task_num}")
    print(f"constraints: {constraints}")

    solution_list = list(p[:] for p in constrained_permutations(task_list, constraints))

    print(f"The number of feasible schedules: {len(solution_list)}")
    best_makespan = np.Inf
    best_schedule=[]
    for i in solution_list:
        temp_jobs = copy.deepcopy(jobs)
        temp_machines = copy.deepcopy(machines)
        temp_makespan = calculate_makespan(i.copy(), jobs=temp_jobs, machines=temp_machines)
        if temp_makespan < best_makespan:
            best_makespan = temp_makespan
            best_schedule = i.copy()
    print(f"optimal solution of job_scheduling: {best_schedule}\nbest_makespan:{best_makespan}")
    return
def quick_check_makespan(jobs=None, machines=None, add_constraints=None):
    task_list = []
    constraints = []
    constraints_idx = 0
    for i in range(len(jobs)):
        constraints.append([])
        for j in range(len(jobs[i])):
            task_list.append(jobs[i][j].task_name)
            constraints[constraints_idx].append(jobs[i][j].task_name)
        constraints_idx += 1
    solution_list = list(p[:] for p in constrained_permutations(task_list, constraints))
    solution_list = np.asarray(solution_list)
    feasible_solutions = np.empty((0,len(solution_list[0])),dtype=np.int8)
    for i in solution_list:
        f = True
        for j in add_constraints:
            temp = np.empty((0,len(j)),dtype=np.int8)
            for k in j:
                temp = np.append(temp, np.where(i==k))
            f = f and (np.all(np.diff(temp) > 0))
        if f:
            feasible_solutions = np.append(feasible_solutions, [i], axis=0)
        

    print(f"number of feasible solutions given constraint: {len(feasible_solutions)}")

    makespans = []
    for i in feasible_solutions:
        temp_jobs = copy.deepcopy(jobs)
        temp_machines = copy.deepcopy(machines)
        makespans.append(calculate_makespan(i.tolist().copy(), jobs=temp_jobs, machines=temp_machines))
    makespans = np.asarray(makespans,dtype=np.float32)
    optimal_solutions = feasible_solutions[np.argmin(makespans, axis=0)]
    print(f"makespans:\n{makespans}")
    print(f"minimum makespan: {makespans.min()}")
    print(f"optimal solutions:\n{optimal_solutions.tolist()}")
    return



# EXECUTION -----------
'''
jobs = [
    [task(1, 1, 5), task(2, 2, 4), task(3, 3, 2)],
    [task(4, 3, 2), task(5, 1, 3), task(6, 2, 7)],
    [task(7, 1, 3), task(8, 2, 6), task(9, 3, 1)]
]
machines = {
    1: machine(),
    2: machine(),
    3: machine()
}

'''

'''
Note:
1. job_scheduling returns an optimal solution and its makespan
2. calculate_makespan returns the makespan of the given solution
3. quick_check_makespan is used to cope with the disjoint graph of the last exercise question,
where  the list "add_constraints" is the order of the tasks on the same machine!!

Warning:
"jobs" is a 2d list storing each task, and the name of each task must be an interger starting from 1 !!
"machines" is a dictionary storing machine objects. Note that the keys of machines must starts from 1 !!


e.g. task(task_name=1, machine=1, processing_time=5), and here the number in "machine=1" must be the same as 
the corresponding key in dict "machines", which means this task shall be assigned to machine 1.


job_scheduling(jobs=copy.deepcopy(jobs), machines=copy.deepcopy(machines))
print(calculate_makespan(solution=[4, 7, 1, 8, 5, 2, 6, 3, 9], jobs=copy.deepcopy(jobs), machines=copy.deepcopy(machines)))
quick_check_makespan(jobs=copy.deepcopy(jobs), machines=copy.deepcopy(machines), add_constraints=[[2,8,6],[1,7,5],[4,3,9]])

'''
jobs = [
    [task(1, 1, 5), task(2, 2, 4), task(3, 3, 2)],
    [task(4, 3, 2), task(5, 1, 3), task(6, 2, 7)],
    [task(7, 1, 3), task(8, 2, 6), task(9, 3, 1)]
]
machines = {
    1: machine(),
    2: machine(),
    3: machine()
}

job_scheduling(jobs=copy.deepcopy(jobs), machines=copy.deepcopy(machines))
print(calculate_makespan(solution=[4, 7, 1, 8, 5, 2, 6, 3, 9], jobs=copy.deepcopy(jobs), machines=copy.deepcopy(machines)))
quick_check_makespan(jobs=copy.deepcopy(jobs), machines=copy.deepcopy(machines), add_constraints=[[8,2,6],[1,7,5],[4,3,9]])
