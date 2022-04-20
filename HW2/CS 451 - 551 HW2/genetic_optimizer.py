import numpy as np
import pandas as pd
from chromosome import Chromosome
import random
from PriorityQueue import PriorityQueue

NUM_POPULATION = 42
data_file_name = "sample_data.csv"
MAX_NUM_ITERATION = 2800

def create_chromosome():
    weights = np.random.uniform(-1, 1, 12)
    return Chromosome(weights)


def init_population(NUM_POPULATION):
    population = []

    for i in range(NUM_POPULATION):
        chromosome = create_chromosome()
        population.append(chromosome)

    return population


def reproduction_chance(chromose_fitness, all_fitnesses):
    return chromose_fitness / sum(all_fitnesses)


def get_data_in_dataframe(filename):
    df = pd.read_csv(filename)
    df = df.drop(columns=df.columns[0])
    return df


def get_x_in_numpy(df):
    X = df.drop(columns=['Y'])
    return X.to_numpy()


def find_fitness(population, X, y_truth):
    for chromosome in population:
        y_pred = chromosome.predict(X)
        chromosome.fitness(y_pred, y_truth)


def get_becoming_parent_chances(population):
    weights = []
    for chromosome in population:
        chance = 1 / chromosome.mse
        weights.append(chance)

    return weights


def is_eligible_parent(candidate_parent, parents):
    if len(parents) % 2 == 1:
        if candidate_parent[0] != parents[-1]:
            return True
        else:
            return False
    else:
        return True


def choose_num_parents(NUM_POPULATION):
    return int(0.75 * NUM_POPULATION) + 1

def select_parents(population, NUM_POPULATION):
    parents = []
    num_iter = choose_num_parents(NUM_POPULATION)
    weights = get_becoming_parent_chances(population)
    first_parent = random.choices(population, weights, k=1)
    parents.append(first_parent[0])

    while len(parents) < num_iter:
        candidate_parent = random.choices(population, weights, k=1)
        if is_eligible_parent(candidate_parent, parents):
            parents.append(candidate_parent[0])

    return parents


def give_birth(male, female):
    crossover_point = random.randint(1, 11)

    first_child_weight = np.concatenate((male.weight[:crossover_point], female.weight[crossover_point:]))
    first_child = Chromosome(first_child_weight)
    second_child_weight = np.concatenate((female.weight[:crossover_point], male.weight[crossover_point:]))
    second_child = Chromosome(second_child_weight)

    return first_child, second_child


def crossover(parents):
    offsprings = []
    for i in range(0, len(parents), 2):
        male = parents[i]
        female = parents[i + 1]

        first_child, second_child = give_birth(male, female)

        offsprings.append(first_child)
        offsprings.append(second_child)

    return offsprings


def mutate(population):
    for chromosome in population:
        for i in range(len(chromosome.weight)):
            chance = random.uniform(0, 1)
            if chance <= 1/14:
                new_weight = random.uniform(-1, 1)
                chromosome.weight[i] = new_weight

    return population

def initial_priority_queue(population):
    pq = PriorityQueue()
    for chromosome in population:
        pq.enqueue(chromosome, chromosome.mse)

    return pq


def find_k_best_individuals(population, NUM_POPULATION, k = 1):
    pq = initial_priority_queue(population)
    k = NUM_POPULATION - choose_num_parents(NUM_POPULATION)
    k_best_indvs = []
    for i in range(k):
        indv, mse = pq.dequeue()
        k_best_indvs.append(indv)

    return k_best_indvs


def get_x_and_y_truth(data_file_name):
    df = get_data_in_dataframe(data_file_name)
    X = get_x_in_numpy(df)
    X = X.transpose()
    y_truth = np.mean(X, axis=0)
    return X, y_truth


def print_log(i):
    if i % 100 == 0:
        l = [round(c.mse, 3) for c in population]
        print(l)
        print(sum(l) / len(l))


if __name__ == '__main__':
    X, y_truth = get_x_and_y_truth(data_file_name)

    population = init_population(NUM_POPULATION)

    for i in range(MAX_NUM_ITERATION):
        find_fitness(population, X, y_truth)
        print_log(i)
        k_best_indvs = find_k_best_individuals(population, NUM_POPULATION)
        parents = select_parents(population, NUM_POPULATION)
        offsprings = crossover(parents)
        offsprings = mutate(offsprings)
        population = offsprings + k_best_indvs
