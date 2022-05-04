import numpy as np
import math
import pandas as pd
import itertools

class Ant_colony():
    def __init__(self, input, alpha, beta, theta, phi, num_of_ants):
        """

        :param input: CSV file name
        :param alpha: construction parameter
        :param beta: construction parameter
        :param theta: heuristic information (For TSP mainly)
        :param phi: evaporate constant
        :param num_of_ants: Number of ants
        """
        self.input = input
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.phi = phi
        self.df_bidirectional = None
        self.num_of_ants = num_of_ants


    def read_graph_from_csv(self):
        df = pd.read_csv(self.input, encoding='unicode_escape')
        return df

    def make_graph(self):
        df = self.read_graph_from_csv()
        num_nodes = df[['node1', 'node2']].max().max() - df[['node1', 'node2']].min().min() + 1
        # print(f"Graph:\n{df}")
        print(f"number of nodes: {num_nodes}\nnumber of edges: {len(df)}")

        # edge
        edge_temp = df[['node1', 'node2']].to_numpy()
        EDGES = np.append(edge_temp, df[['node2', 'node1']].to_numpy(), axis=0)
        # distance
        distance_temp = df['distance'].to_numpy()
        distances = np.append(distance_temp, df['distance'].to_numpy())
        # pheromone
        pheromone_temp = df['pheromone'].to_numpy()
        pheromones = np.append(pheromone_temp, df['pheromone'].to_numpy())

        # bidirectional graph
        df_bidirectional = pd.DataFrame(
            {'node1': EDGES[:, 0], 'node2': EDGES[:, 1], 'distances': distances, 'pheromone': pheromones})
        df_bidirectional = df_bidirectional.drop_duplicates(ignore_index=True)
        # print(df_bidirectional)
        self.df_bidirectional = df_bidirectional


    def transition_probability(self, start_point, intermediate_solution, mode):
        # self.make_graph()
        total_pheromone = 0.0
        tp_dict = {}
        # total
        for ind in range(len(self.df_bidirectional)):
            if self.df_bidirectional.loc[ind, "node1"] == start_point:
                # print(self.df_bidirectional["node1"][ind], self.df_bidirectional["node2"][ind])
                try:
                    if self.df_bidirectional.loc[ind, "node2"] != intermediate_solution[0] and self.df_bidirectional.loc[ind, "node2"] != intermediate_solution[1]:
                        total_pheromone += ((self.df_bidirectional.loc[ind,"pheromone"])**self.alpha) * (self.theta**self.beta)

                except: # when there is no intermediate_solution
                    total_pheromone += ((self.df_bidirectional.loc[ind,"pheromone"])**self.alpha) * (self.theta**self.beta)
        if mode:
            print(f"Total Pheromone is: {total_pheromone}")

        # calculate pheromone for each
        for ind in range(len(self.df_bidirectional)):
            if self.df_bidirectional.loc[ind, "node1"] == start_point:
                # print(self.df_bidirectional["node1"][ind], self.df_bidirectional["node2"][ind])
                try:
                    if self.df_bidirectional.loc[ind, "node2"] != intermediate_solution[0] and self.df_bidirectional.loc[
                        ind, "node2"] != intermediate_solution[1]:
                        transition_probability = (((self.df_bidirectional.loc[ind,"pheromone"])**self.alpha) * (self.theta**self.beta))/ total_pheromone
                        tp_dict[f"{start_point}_{self.df_bidirectional.loc[ind, 'node2']}"] = transition_probability
                        if mode:
                            print(f"p_{start_point}_{self.df_bidirectional.loc[ind, 'node2']}(t)={transition_probability}")
                except:
                    transition_probability =  (((self.df_bidirectional.loc[ind,"pheromone"])**self.alpha) * (self.theta**self.beta)) / total_pheromone
                    tp_dict[f"{start_point}_{self.df_bidirectional.loc[ind, 'node2']}"] = transition_probability
                    if mode:
                        print(f"p_{start_point}_{self.df_bidirectional.loc[ind, 'node2']}(t)={transition_probability}")
        max_index = np.argmax(list(tp_dict.values()))
        min_index = np.argmin(list(tp_dict.values()))
        if mode:
            print(f"Maximum pheromone is path: {list(tp_dict.keys())[max_index]}")
            print(f"Minimum pheromone is path: {list(tp_dict.keys())[min_index]}")
        return total_pheromone, tp_dict
    def path_from_point(self, starting_point, ending_point, highest):

        df_temp = {}
        walked = []
        step = 1
        while True:
            max_pheromone = 0.0
            min_pheromone = 10000.0
            print(f"===============Step Number {step}===============")
            for ind in range(len(self.df_bidirectional)):
                if self.df_bidirectional.loc[ind, "node1"] == starting_point and self.df_bidirectional.loc[ind, "node2"] not in walked:
                    if max_pheromone < self.df_bidirectional.loc[ind, 'pheromone']:
                        max_pheromone = self.df_bidirectional.loc[ind, 'pheromone']
                    if min_pheromone > self.df_bidirectional.loc[ind, 'pheromone']:
                        min_pheromone = self.df_bidirectional.loc[ind, 'pheromone']
                if highest:
                    if self.df_bidirectional.loc[ind, "node1"] == starting_point and self.df_bidirectional.loc[ind, 'pheromone']==max_pheromone:
                        df_temp["end_point"] =self.df_bidirectional.loc[ind, "node2"]
                else:
                    if self.df_bidirectional.loc[ind, "node1"] == starting_point and self.df_bidirectional.loc[
                        ind, 'pheromone'] == min_pheromone:
                        df_temp["end_point"] = self.df_bidirectional.loc[ind, "node2"]
            if highest:
                walked.append(starting_point)
                print(f"The maximum pheromone for the next step is {max_pheromone}")
                print(f"Path indication: {starting_point} --> {df_temp['end_point']}")
            else:
                walked.append(starting_point)
                print(f"The minimum pheromone for the next step is {min_pheromone}")
                print(f"Path indication: {starting_point} --> {df_temp['end_point']}")

            print(f"Path walked: {walked}")
            if starting_point == df_temp["end_point"]:
                print("Next route will be repeated")
                break
            if ending_point == df_temp["end_point"]:
                walked.append(ending_point)
                print(f"Reached destination, Final Route: {walked}")
                return walked
            # update for next round
            starting_point = df_temp['end_point']
            step += 1
        return walked

    def value_of_concentration(self, edge, Q):
        _, transition_prob_dic = self.transition_probability(start_point=edge[0], intermediate_solution=[], mode=False)
        print("============Process started below============")
        f_x_k_t = 0.0
        current_pheromone = 0.0
        edge_transition_prob = transition_prob_dic[f"{edge[0]}_{edge[1]}"]
        print(f"Transition probability for the edge: {edge_transition_prob}")
        average_number_of_ants = self.num_of_ants * edge_transition_prob
        print(f"Average number of ants using the edge: {average_number_of_ants}")
        for ind in range(len(self.df_bidirectional)):
            if self.df_bidirectional.loc[ind, "node1"] == edge[0] and self.df_bidirectional.loc[ind, "node2"] == edge[1]:
                current_pheromone = self.df_bidirectional.loc[ind, "pheromone"]
                # quality of the solution -->edge distance
                f_x_k_t = self.df_bidirectional.loc[ind, "distances"]
        print(f"f_x_k_t: {f_x_k_t}")
        delta_pheromone_update = Q / f_x_k_t
        print(f"Delta pheromone update: {delta_pheromone_update}")
        # evaporation (where should i place this)
        evaported_pheromone = (1- self.phi) * current_pheromone

        # update to original
        new_pheromone = evaported_pheromone + average_number_of_ants * delta_pheromone_update

        print(f"tao_{edge[0]},{edge[1]}_(t+1)= {new_pheromone}")
        return new_pheromone






def main():

    input = "ant.csv"
    alpha = 1
    beta = 1
    theta = 1 # heuristic information associated with edge (i, j),, theta_ij = 1 / c_ij
    phi = 0.2 # evaporate constant
    num_of_ants = 10000
    ant = Ant_colony(input=input, alpha=alpha, beta= beta, theta= theta, phi=phi, num_of_ants=num_of_ants)
    ant.make_graph()
    ant.transition_probability(start_point=3, intermediate_solution=[1, 3], mode=True) # if mode, print out
    ant.path_from_point(starting_point= 1, ending_point=5, highest=True)
    ant.value_of_concentration(edge=[1,5], Q = 0.01)


main()