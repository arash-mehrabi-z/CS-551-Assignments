import numpy as np
import pandas as pd
from chromosome import Chromosome
import random
from PriorityQueue import PriorityQueue
import matplotlib.pyplot as plt   # plotting

NUM_POPULATION = 28
data_file_name = "sample_data.csv"
MAX_NUM_ITERATION = 2800

best_ind_per_generation = []
mean_mse_per_generation = []
plt.figure(0, figsize=(16, 12))
plt.figure(1, figsize=(16, 12))

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
    num_parents = int(0.95 * NUM_POPULATION)
    if num_parents % 2 != 0:
        num_parents += 1
    return num_parents

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
            if chance <= 1/18:
                new_weight = random.uniform(-1, 1)
                chromosome.weight[i] = new_weight

    return population


def initial_priority_queue(population):
    pq = PriorityQueue()
    for chromosome in population:
        pq.enqueue(chromosome, chromosome.mse)

    return pq


def choose_k(NUM_POPULATION):
    return NUM_POPULATION - choose_num_parents(NUM_POPULATION)


def find_k_best_individuals(population, NUM_POPULATION, k = 1):
    pq = initial_priority_queue(population)
    k = choose_k(NUM_POPULATION)
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
    if i % 500 == 0 or i < 10:
        chromosome_mse = [round(chromosome.mse, 3) for chromosome in population]
        print("Generation: ", str(i))
        print("Error values of chromosomes of this generation is like following:")
        print(chromosome_mse)
        print("The average is: ", sum(chromosome_mse) / len(chromosome_mse))
        print("-----")


def plot(x_coordinate, y_coordinate, xlabel, ylabel, title, save_file_name):
    plt.figure(figsize=(16, 12))
    plt.plot(x_coordinate, y_coordinate)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_file_name)


def plot_each_chromosome(xlabel, ylabel, title, save_file_name):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_file_name)


def plot_results(best_ind_per_generation, mean_mse_per_generation):
    plt.figure(0)
    plot_each_chromosome("Generation", "Error Value (MSE)",
      "Error Value of Each Chromosome of Each Generation (Same color means same generation)",
      "each_chromose.png")

    plt.figure(1)
    plot_each_chromosome("Generation", "Error Value (MSE)",
      "Error Value of Each Chromosome of Each Generation (Every 400 Generations)",
      "each_chromose_every_400.png")

    x_coordinate = [i for i in range(MAX_NUM_ITERATION)]
    y_coordinate = [chromosome.mse for chromosome in best_ind_per_generation]

    plot(x_coordinate, y_coordinate, "Generation", "Error Value (MSE)",
         "Error Value of Best Individual of Each Generation", "best_indv_mse.png")

    plot(x_coordinate[:14], y_coordinate[:14], "Generation", "Error Value (MSE)",
         "Error Value of Best Individual of Each Generation (First 14 Generations)", "best_indv_mse_0_14.png")

    plot(x_coordinate[1000:1014], y_coordinate[1000:1014], "Generation", "Error Value (MSE)",
         "Error Value of Best Individual of Each Generation (Generations 1000 - 1014)", "best_indv_mse_1000_1014.png")

    y_coordinate = mean_mse_per_generation
    plot(x_coordinate, y_coordinate, "Generation", "Mean MSE value",
         "Mean MSE value of Each Generation", "mean_mse_per_generation.png")
    plot(x_coordinate[:100], y_coordinate[:100], "Generation", "Mean MSE value",
         "Mean MSE value of Each Generation (First 100 Generations)", "mean_mse_per_generation_0_100.png")
    plot(x_coordinate[1000:1100], y_coordinate[1000:1100], "Generation", "Mean MSE value",
         "Mean MSE value of Each Generation (Generations 1000 - 1100)", "mean_mse_per_generation_1000_1100.png")


def calculate_mean_mse(population):
    sum_mse = 0
    for chromosome in population:
        sum_mse += chromosome.mse

    return sum_mse / len(population)


def record_results(population, k_best_indvs, i):
    print_log(i)
    best_ind_per_generation.append(k_best_indvs[0])
    mean_mse = calculate_mean_mse(population)
    mean_mse_per_generation.append(mean_mse)
    population_mse = [chromosome.mse for chromosome in population]
    if i % 400 == 0:
        plt.figure(1)
        plt.scatter([i] * len(population_mse), population_mse)

    plt.figure(0)
    plt.scatter([i] * len(population_mse), population_mse)


if __name__ == '__main__':
    X, y_truth = get_x_and_y_truth(data_file_name)

    population = init_population(NUM_POPULATION)
    for i in range(MAX_NUM_ITERATION):
        find_fitness(population, X, y_truth)
        k_best_indvs = find_k_best_individuals(population, NUM_POPULATION)
        record_results(population, k_best_indvs, i)
        parents = select_parents(population, NUM_POPULATION)
        offsprings = crossover(parents)
        offsprings = mutate(offsprings)
        population = offsprings + k_best_indvs

    plot_results(best_ind_per_generation, mean_mse_per_generation)

