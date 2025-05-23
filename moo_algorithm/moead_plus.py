import multiprocessing
import sys
import os
import numpy as np
# Add the parent directory to the module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from moo_algorithm.metric import cal_hv_front
from population import Population, Individual
from graph.graph import Graph

def init_weight_vectors_2d(pop_size):
    wvs = []
    for i in np.arange(0, 1 + sys.float_info.epsilon, 1 / (pop_size - 1)):
        wvs.append([i, 1 - i])
    return np.array(wvs)

def init_weight_vectors_3d(pop_size):
    wvs = []
    for i in np.arange(0, 1 + sys.float_info.epsilon, 1 / (pop_size - 1)):
        for j in np.arange(0, 1 + sys.float_info.epsilon, 1 / (pop_size - 1)):
            if i + j <= 1:
                wvs.append([i, j, 1 - i - j])
    return np.array(wvs)

def init_weight_vectors_3d_plus(pop_size):
    step = 1 / (pop_size - 1)
    wvs = [
        [i, j, 1 - i - j]
        for i in np.arange(0, 1 + step, step)
        for j in np.arange(0, 1 - i + step, step)  # Ensures i + j <= 1
    ]
    return np.array(wvs)

def init_weight_vectors_4d(pop_size):
    wvs = []
    for i in np.arange(0, 1 + sys.float_info.epsilon, 1 / (pop_size - 1)):
        for j in np.arange(0, 1 + sys.float_info.epsilon, 1 / (pop_size - 1)):
            for k in np.arange(0, 1 + sys.float_info.epsilon, 1 / (pop_size - 1)):
                if i + j + k <= 1:
                    wvs.append([i, j, k, 1 - i - j - k])
    return np.array(wvs)


class MOEADPopulation(Population):
    def __init__(self, pop_size,  neighborhood_size, init_weight_vectors):
        super().__init__(pop_size)
        self.neighborhood_size = neighborhood_size
        self.external_pop = []
        self.weights = init_weight_vectors(self.pop_size)
        self.neighborhoods = self.init_neighborhood()
        self.objectives_tuple = set()

    def init_neighborhood(self):
        B = np.empty([self.pop_size, self.neighborhood_size], dtype=int)
        for i in range(self.pop_size):
            wv = self.weights[i]
            euclidean_distances = np.empty([self.pop_size], dtype=float)
            for j in range(self.pop_size):
                euclidean_distances[j] = np.linalg.norm(wv - self.weights[j])
            B[i] = np.argsort(euclidean_distances)[:self.neighborhood_size]
        return B

    def reproduction(self, problem, crossover_operator, mutation_operator):
        offspring = []
        for i in range(self.pop_size):
            parent1, parent2 = np.random.choice(self.neighborhoods[i].tolist(), 2, replace=False)
            off1, off2 = crossover_operator(problem, self.indivs[parent1], self.indivs[parent2])
            if np.random.rand() < 0.15:
                off1 = mutation_operator(problem, off1)
            offspring.append(off1)
        return offspring
    
    def mutation(self, problem, mutation_operator):
        for i in range(self.pop_size):
            if np.random.rand() < 0.1:
                self.indivs[i] = mutation_operator(problem, self.indivs[i])
    

    def natural_selection(self):
        self.indivs, O = self.indivs[:self.pop_size], self.indivs[self.pop_size:]
        for i in range(self.pop_size):
            indi = O[i]
            wv = self.weights[i]
            value_indi = np.sum(wv * indi.objectives)
            for j in self.neighborhoods[i]:
                if value_indi < np.sum(wv * self.indivs[j].objectives):
                    self.indivs[j] = indi

    # def update_external(self, indivs: list):
    #     for indi in indivs:
    #         old_size = len(self.external_pop)
    #         self.external_pop = [other for other in self.external_pop
    #                              if not indi.dominates(other)]
    #         if old_size > len(self.external_pop):
    #             self.external_pop.append(indi)
    #             continue
    #         for other in self.external_pop:
    #             if other.dominates(indi):
    #                 break
    #         else:
    #             self.external_pop.append(indi)


    def update_external(self, indivs: list):
        for indi in indivs:
            if tuple(indi.objectives) in self.objectives_tuple:
                continue
            # old_size = len(self.external_pop)
            self.external_pop = [other for other in self.external_pop
                                if not indi.dominates(other)]
            # if old_size > len(self.external_pop):
            #     self.external_pop.append(indi)
            #     continue
            for other in self.external_pop:
                if other.dominates(indi):
                    break
            else:
                self.external_pop.append(indi)
                self.objectives_tuple.add(tuple(indi.objectives))
            
            # Debug: Check for duplicate chromosomes

            # chromosomes = [tuple(ind.chromosome) for ind in self.external_pop]
            # if len(chromosomes) != len(set(chromosomes)):
            #     print("Duplicate chromosome detected in external population!")
            #     print("Chromosome:", indi.chromosome)
            #     raise ValueError("Duplicate chromosome detected in external population!")
    
    def filter_external(self):
        objectives = set()
        new_external_pop = []
        for indi in self.external_pop:
            if tuple(indi.objectives) not in objectives:
                new_external_pop.append(indi)
                objectives.add(tuple(indi.objectives))
        self.external_pop = new_external_pop


    
    # def update_weights(self, problem, indivs: list):
    #     for i in range(self.pop_size):
    #         wv = self.weights[i]
    #         self.indivs[i].objectives = problem.evaluate(indivs[i].chromosome)
    #         value_indi = np.sum(wv * self.indivs[i].objectives)
    #         for j in self.neighborhoods[i]:
    #             if value_indi < np.sum(wv * self.indivs[j].objectives):
    #                 self.indivs[j] = self.indivs[i]


def run_moead_plus(processing_number, problem, indi_list, pop_size, max_gen, neighborhood_size, 
              init_weight_vectors, crossover_operator, mutation_operator, cal_fitness):
    np.random.seed(0)
    moead_pop = MOEADPopulation(pop_size, neighborhood_size, init_weight_vectors)
    moead_pop.pre_indi_gen(indi_list)

    pool = multiprocessing.Pool(processing_number)
    arg = []
    for individual in moead_pop.indivs:
        arg.append((problem, individual))
    result = pool.starmap(cal_fitness, arg)
    for individual, fitness in zip(moead_pop.indivs, result):
        individual.objectives = fitness
    
    moead_pop.update_external(moead_pop.indivs)
    moead_pop.filter_external()
    # moead_pop.update_weights(problem, moead_pop.indivs)

    # print("Generation 0: ", cal_hv_front(moead_pop.external_pop, np.array([1, 1, 1])))

    history = {}
    Pareto_store = []
    for indi in moead_pop.external_pop:
        Pareto_store.append(list(indi.objectives))
    history[0] = Pareto_store

    for gen in range(max_gen):
        offspring = moead_pop.reproduction(problem, crossover_operator, mutation_operator)
        arg = []
        for individual in offspring:
            arg.append((problem, individual))
        result = pool.starmap(cal_fitness, arg)
        for individual, fitness in zip(offspring, result):
            individual.objectives = fitness
        moead_pop.update_external(offspring)
        moead_pop.filter_external()
        moead_pop.indivs.extend(offspring)
        # moead_pop.update_weights(problem, offspring)
        # print("Generation {}: ".format(gen + 1), cal_hv_front(moead_pop.external_pop, np.array([1, 1, 1])))
        moead_pop.natural_selection()

        Pareto_store = []
        for indi in moead_pop.external_pop:
            Pareto_store.append(list(indi.objectives))
        history[gen+1] = Pareto_store
    pool.close()

    # for i in moead_pop.external_pop:
    #     print(i.objectives)

    # print("Final:" , cal_hv_front(moead_pop.external_pop, np.array([1, 1, 1])))

    # return cal_hv_front(moead_pop.external_pop, np.array([1, 1, 1]))
    # return moead_pop.external_pop

    # list = []
    # for i in moead_pop.external_pop:
    #     list.append(i.objectives)
    # return list


    return history


if __name__ == "__main__":
    from utils_new import crossover_operator, mutation_operator, calculate_fitness, create_individual_pickup
    filepath = '.\\data\\dpdptw\\200\\LC1_2_1.csv'
    graph = Graph(filepath)
    indi_list = [create_individual_pickup(graph) for _ in range(100)]
    run_moead_plus(4, graph, indi_list, 100, 100, 10, init_weight_vectors_4d, crossover_operator,mutation_operator, calculate_fitness)
